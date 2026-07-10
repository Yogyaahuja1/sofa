"""
Membrane (flat surface) PINN training script.

Flat membrane: 1031 vertices, TriangularFEM, 2D strain/stress.
Column layout (per vertex i):
  dx/dy/dz{i}       — current deformation
  pdx/pdy/pdz{i}    — previous deformation
  fvx/fvy/fvz{i}    — contactProxy = freePos - curPos
  sax/say/saz{i}    — accumulated contact stress EMA (α=0.9)
  msxx/msyy/msxy{i} — TriangularFEM stress (2D, 3 components)
  mexx/meyy/mexy{i} — TriangularFEM strain (2D, 3 components)

Usage examples:
  python train_membrane_pinn.py --run-id membrane_v1
  python train_membrane_pinn.py --run-id membrane_v2 --no-strain --use-stress
  python train_membrane_pinn.py --run-id membrane_v3 --arch dual_attn
"""

import os as _os
import sys
SOFA_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pinn_model import LiverSeqAttnFlex, LiverDualAttnFlex

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--run-id', type=str, default=None)
parser.add_argument('--n-lags', type=int, default=8)
parser.add_argument('--n-neighbours', type=int, default=20)
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--sustain-beta', type=float, default=5.0)
parser.add_argument('--arch', type=str, default='dual_attn',
                    choices=['seq_attn', 'dual_attn'])
# feature toggles
parser.add_argument('--no-accstress', action='store_true', help='Exclude accStress EMA from nb features')
parser.add_argument('--no-strain',    action='store_true', help='Exclude 2D strain (mexx/meyy/mexy)')
parser.add_argument('--use-stress',   action='store_true', help='Include 2D FEM stress (msxx/msyy/msxy)')
parser.add_argument('--use-proxy',    action='store_true', help='Include contactProxy (fvx/fvy/fvz) per vertex')
# output mode
parser.add_argument('--force-only', action='store_true', default=True,
                    help='Predict only force (3 outputs), skip deformation head')
parser.add_argument('--no-force-only', dest='force_only', action='store_false')
parser.add_argument('--filter-short', action='store_true', default=True,
                    help='Drop sessions with fewer than 30 rows')
parser.add_argument('--no-filter-short', dest='filter_short', action='store_false')
# loss
parser.add_argument('--high-force-gamma', type=float, default=2.0,
                    help='Power for force-magnitude loss weighting')
parser.add_argument('--csv-path', type=str, default=None,
                    help='Override default CSV (default: data/flat_surface_training_data.csv)')
args = parser.parse_args()

import glob as _glob
def _auto_run_id():
    existing = _glob.glob(f'{SOFA_ROOT}/pinn_project/train/membrane_v*.pth')
    nums = []
    for p in existing:
        base = p.split('_v')[-1].replace('.pth','')
        if base.isdigit(): nums.append(int(base))
    return f'v{max(nums)+1 if nums else 1}'
RUN_ID = args.run_id if args.run_id else _auto_run_id()
N_LAGS        = args.n_lags
N_NEIGHBOURS  = args.n_neighbours
N_EPOCHS      = args.epochs
SUSTAIN_BETA  = args.sustain_beta
ARCH          = args.arch
USE_ACCSTRESS = not args.no_accstress
USE_STRAIN    = not args.no_strain
USE_STRESS    = args.use_stress
USE_PROXY     = args.use_proxy
FORCE_ONLY    = args.force_only
FILTER_SHORT  = args.filter_short
HIGH_FORCE_GAMMA = args.high_force_gamma
CSV_PATH_OVERRIDE = args.csv_path

N_STEPS = N_LAGS
n_nb_feat = 3 + 3*USE_ACCSTRESS + 3*USE_STRAIN + 3*USE_STRESS + 3*USE_PROXY
print(f"\nRun: {RUN_ID}  [MEMBRANE]")
print(f"arch={ARCH}  n_steps={N_STEPS}  K={N_NEIGHBOURS}  n_nb_feat={n_nb_feat}")
print(f"features: deform(always) accstress={USE_ACCSTRESS} strain(2D)={USE_STRAIN} "
      f"stress(2D)={USE_STRESS} proxy={USE_PROXY}")
print(f"high_force_gamma={HIGH_FORCE_GAMMA}\n")

CSV_PATH     = CSV_PATH_OVERRIDE if CSV_PATH_OVERRIDE else f'{SOFA_ROOT}/pinn_project/data/flat_surface_training_data.csv'
RESULTS_FILE = f'{SOFA_ROOT}/pinn_project/train/membrane_results.txt'
N_VERTICES   = 1031
FIXED_INDICES = []
BATCH_SIZE   = 64
LR           = 3e-4
TRAIN_SPLIT  = 0.8
W_FORCE      = 0.5
W_DEFORM     = 1.0
SEED         = 42
EARLY_STOP   = 800

torch.manual_seed(SEED); np.random.seed(SEED)

# ── Load + clean ───────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(CSV_PATH, dtype={c: 'float32' for c in pd.read_csv(CSV_PATH, nrows=0).columns
                                  if c not in ('step', 'session_id')})
df = df.replace([np.inf, -np.inf], np.nan).dropna()

sess_counts = df.groupby('session_id')['session_id'].transform('count')
df = df[sess_counts >= N_STEPS * 3].reset_index(drop=True)

if FILTER_SHORT:
    sess_counts2 = df.groupby('session_id')['session_id'].transform('count')
    before = len(df)
    df = df[sess_counts2 >= 30].reset_index(drop=True)
    print(f"filter-short: dropped {before - len(df)} rows from sessions < 30 rows")

max_disp = df[[c for c in df.columns if c.startswith('dx') or c.startswith('dy') or c.startswith('dz')]].values
pdmax    = df[[c for c in df.columns if c.startswith('pdx') or c.startswith('pdy') or c.startswith('pdz')]].values
df = df[np.abs(max_disp - pdmax).max(axis=1) < 2.0].reset_index(drop=True)

df['tool_fx'] = df['tool_fx'].clip(-100.0, 100.0)
df['tool_fy'] = df['tool_fy'].clip(-100.0, 100.0)
df['tool_fz'] = df['tool_fz'].clip(-100.0, 100.0)

N_rows = len(df)
print(f"Rows after cleaning: {N_rows}")

# ── Sustained-contact flag ─────────────────────────────────────────────────────
session_ids = df['session_id'].values
df_fmag = np.linalg.norm(df[['tool_fx', 'tool_fy', 'tool_fz']].values, axis=1)
is_sustained = np.zeros(N_rows, dtype=np.float32)
for sid in np.unique(session_ids):
    if sid == 0: continue
    idx = df.index[df['session_id'] == sid].values
    high = df_fmag[idx] > 3.0
    run_id_arr = np.zeros(len(high), dtype=int)
    cur = 0
    for i in range(len(high)):
        cur = cur + 1 if high[i] else 0
        run_id_arr[i] = cur
    if run_id_arr.max() >= 10:
        is_sustained[idx[high]] = 1.0
print(f"Sustained-contact rows: {int(is_sustained.sum())} / {N_rows} "
      f"({100*is_sustained.sum()/N_rows:.1f}%)")

# ── Load vertex positions + per-vertex arrays ──────────────────────────────────
vertex_pos = np.load(f'{SOFA_ROOT}/pinn_project/data/membrane_vertices.npy')  # (1031,3)
tool_pos   = df[['tool_x', 'tool_y', 'tool_z']].values
real_time  = df['real_time'].values
sim_time   = df['sim_time'].values

NV = N_VERTICES
dx_all  = df[[f'dx{i}'  for i in range(NV)]].values
dy_all  = df[[f'dy{i}'  for i in range(NV)]].values
dz_all  = df[[f'dz{i}'  for i in range(NV)]].values
pdx_all = df[[f'pdx{i}' for i in range(NV)]].values
pdy_all = df[[f'pdy{i}' for i in range(NV)]].values
pdz_all = df[[f'pdz{i}' for i in range(NV)]].values
ddx_all = dx_all - pdx_all
ddy_all = dy_all - pdy_all
ddz_all = dz_all - pdz_all

if USE_ACCSTRESS:
    sax_all = df[[f'sax{i}' for i in range(NV)]].values
    say_all = df[[f'say{i}' for i in range(NV)]].values
    saz_all = df[[f'saz{i}' for i in range(NV)]].values

if USE_STRAIN:
    mexx_all = df[[f'mexx{i}' for i in range(NV)]].values
    meyy_all = df[[f'meyy{i}' for i in range(NV)]].values
    mexy_all = df[[f'mexy{i}' for i in range(NV)]].values

if USE_STRESS:
    msxx_all = df[[f'msxx{i}' for i in range(NV)]].values
    msyy_all = df[[f'msyy{i}' for i in range(NV)]].values
    msxy_all = df[[f'msxy{i}' for i in range(NV)]].values

if USE_PROXY:
    fvx_all = df[[f'fvx{i}' for i in range(NV)]].values
    fvy_all = df[[f'fvy{i}' for i in range(NV)]].values
    fvz_all = df[[f'fvz{i}' for i in range(NV)]].values

tool_x_all  = df['tool_x'].values;  tool_y_all  = df['tool_y'].values
tool_z_all  = df['tool_z'].values
tool_vx_all = df['tool_vx'].values; tool_vy_all = df['tool_vy'].values
tool_vz_all = df['tool_vz'].values
tool_fx_all = df['tool_fx'].values; tool_fy_all = df['tool_fy'].values
tool_fz_all = df['tool_fz'].values

def log1p_signed(x): return np.sign(x) * np.log1p(np.abs(x))

# ── KNN neighbours (by mesh vertex distance, not deformed position) ────────────
print("Computing KNN neighbours...")
neighbour_idx = np.zeros((N_rows, N_NEIGHBOURS), dtype=np.int32)
min_dist_to_mesh = np.zeros(N_rows, dtype=np.float32)
for row in range(N_rows):
    dists = np.linalg.norm(vertex_pos - tool_pos[row], axis=1)
    idx_s = np.argsort(dists)
    neighbour_idx[row]    = idx_s[:N_NEIGHBOURS]
    min_dist_to_mesh[row] = dists[idx_s[0]]

# ── Global features ────────────────────────────────────────────────────────────
dt_arr  = np.diff(real_time, prepend=real_time[0])
dt_arr[session_ids != np.concatenate([[session_ids[0]], session_ids[:-1]])] = 0.0
dt_cum  = np.stack([dt_arr * (k+1) for k in range(N_STEPS)], axis=1).astype(np.float32)
dt_pred = dt_arr.reshape(-1, 1).astype(np.float32)

current_pos_vel = np.stack([tool_x_all, tool_y_all, tool_z_all,
                             tool_vx_all, tool_vy_all, tool_vz_all], axis=1).astype(np.float32)
delta_min_dist  = np.diff(min_dist_to_mesh, prepend=min_dist_to_mesh[0]).astype(np.float32)
lag1_contact    = np.zeros(N_rows, dtype=np.float32)
lag1_contact[1:] = min_dist_to_mesh[:-1]
contact_features = np.stack([min_dist_to_mesh, delta_min_dist, lag1_contact], axis=1)

accel = np.zeros((N_rows, 3), dtype=np.float32)
for row in range(N_rows):
    s1, s3 = max(0, row-1), max(0, row-3)
    if session_ids[s1] != session_ids[row] or session_ids[s3] != session_ids[row]: continue
    dt = real_time[s1] - real_time[s3]
    if dt < 1e-4: continue
    accel[row, 0] = (tool_vx_all[s1] - tool_vx_all[s3]) / dt
    accel[row, 1] = (tool_vy_all[s1] - tool_vy_all[s3]) / dt
    accel[row, 2] = (tool_vz_all[s1] - tool_vz_all[s3]) / dt
accel = np.clip(accel, -500.0, 500.0)

n_global = N_STEPS + 13
X_global = np.concatenate([dt_cum, dt_pred, current_pos_vel,
                            contact_features, accel], axis=1).astype(np.float32)
assert X_global.shape[1] == n_global, f"Expected {n_global} global features, got {X_global.shape[1]}"

# ── Lag features ───────────────────────────────────────────────────────────────
lag_w = 9 + N_NEIGHBOURS * n_nb_feat
print(f"Building lag features ({N_STEPS} steps, {lag_w} per step, {n_global} global)...")
X_lags = np.zeros((N_rows, N_STEPS * lag_w), dtype=np.float32)

run_start_idx = np.zeros(N_rows, dtype=np.int64)
for row in range(1, N_rows):
    run_start_idx[row] = row if session_ids[row] != session_ids[row-1] else run_start_idx[row-1]

for row in range(N_rows):
    nb = neighbour_idx[row]
    src_rows   = [max(0, row - (k+1)) for k in range(N_STEPS)]
    sess_start = run_start_idx[row]
    for k, src in enumerate(src_rows):
        off = k * lag_w
        same_sess = (session_ids[src] == session_ids[row])
        if not same_sess:
            src = sess_start
        X_lags[row, off]   = tool_x_all[src]
        X_lags[row, off+1] = tool_y_all[src]
        X_lags[row, off+2] = tool_z_all[src]
        X_lags[row, off+3] = tool_vx_all[src] if same_sess else 0.0
        X_lags[row, off+4] = tool_vy_all[src] if same_sess else 0.0
        X_lags[row, off+5] = tool_vz_all[src] if same_sess else 0.0
        X_lags[row, off+6] = log1p_signed(tool_fx_all[src]) if same_sess else 0.0
        X_lags[row, off+7] = log1p_signed(tool_fy_all[src]) if same_sess else 0.0
        X_lags[row, off+8] = log1p_signed(tool_fz_all[src]) if same_sess else 0.0
        nb_off = off + 9
        for j, nv in enumerate(nb):
            feat = [ddx_all[src, nv], ddy_all[src, nv], ddz_all[src, nv]] if same_sess else [0., 0., 0.]
            if USE_ACCSTRESS:
                feat += ([sax_all[src, nv], say_all[src, nv], saz_all[src, nv]] if same_sess else [0., 0., 0.])
            if USE_STRAIN:
                feat += ([mexx_all[src, nv], meyy_all[src, nv], mexy_all[src, nv]] if same_sess else [0., 0., 0.])
            if USE_STRESS:
                feat += ([msxx_all[src, nv], msyy_all[src, nv], msxy_all[src, nv]] if same_sess else [0., 0., 0.])
            if USE_PROXY:
                feat += ([fvx_all[src, nv], fvy_all[src, nv], fvz_all[src, nv]] if same_sess else [0., 0., 0.])
            X_lags[row, nb_off + j*n_nb_feat : nb_off + (j+1)*n_nb_feat] = feat

print("Lag features built.")
X_combined = np.concatenate([X_global, X_lags], axis=1)
N_INPUTS = X_combined.shape[1]
print(f"Total inputs: {N_INPUTS}")

# ── Targets ────────────────────────────────────────────────────────────────────
active_verts = list(range(N_VERTICES))  # no fixed vertices on membrane
dy_cols, pdy_cols = [], []
for i in active_verts:
    dy_cols  += [f'dx{i}', f'dy{i}', f'dz{i}']
    pdy_cols += [f'pdx{i}', f'pdy{i}', f'pdz{i}']
Y_deform  = (df[dy_cols].values - df[pdy_cols].values).astype(np.float32)
force_raw = df[['tool_fx', 'tool_fy', 'tool_fz']].values.astype(np.float32)
del df; import gc; gc.collect()
Y_force   = np.sign(force_raw) * np.log1p(np.abs(force_raw))
N_FORCE   = 3
if FORCE_ONLY:
    Y_combined = Y_force.astype(np.float32)
    print("force-only mode: predicting 3 force outputs only")
else:
    Y_combined = np.concatenate([Y_force, Y_deform], axis=1).astype(np.float32)
N_OUT = Y_combined.shape[1]
print(f"Total outputs: {N_OUT}")

# ── Train/val split (row-balanced session-level) ──────────────────────────────
rng = np.random.RandomState(SEED)
all_sids     = np.unique(session_ids)
contact_sids = all_sids[all_sids != 0]
rng.shuffle(contact_sids)
sid_sizes    = {sid: int((session_ids == sid).sum()) for sid in contact_sids}
total_contact = sum(sid_sizes.values())
target_val   = (1 - TRAIN_SPLIT) * total_contact
fillable = [sid for sid in contact_sids if sid_sizes[sid] <= target_val]
test_sids, val_rows = set(), 0
for sid in fillable:
    if val_rows < target_val:
        test_sids.add(sid); val_rows += sid_sizes[sid]
print(f"Held-out sessions ({len(test_sids)}/{len(contact_sids)}): {sorted(test_sids)}")

is_test = np.array([sid in test_sids for sid in session_ids])
zero_idx = np.where(session_ids == 0)[0]
rng.shuffle(zero_idx)
n_zero_test = int(round((1-TRAIN_SPLIT) * len(zero_idx)))
for i in zero_idx[:n_zero_test]:
    is_test[i] = True

train_idx = torch.LongTensor(np.where(~is_test)[0])
test_idx  = torch.LongTensor(np.where(is_test)[0])

X_t = torch.FloatTensor(X_combined)
Y_t = torch.FloatTensor(Y_combined)
S_t = torch.FloatTensor(is_sustained)

X_train_raw = X_t[train_idx]; X_test_raw = X_t[test_idx]
Y_train_raw = Y_t[train_idx]; Y_test_raw = Y_t[test_idx]
S_train_raw = S_t[train_idx]; S_test_raw = S_t[test_idx]

perm = torch.randperm(len(X_train_raw))
X_train_raw = X_train_raw[perm]; Y_train_raw = Y_train_raw[perm]; S_train_raw = S_train_raw[perm]
print(f"Train: {len(X_train_raw)}  |  Val: {len(X_test_raw)}")

# Normalise
Y_mean = Y_train_raw.mean(0); Y_std = Y_train_raw.std(0) + 1e-8
Y_train = (Y_train_raw - Y_mean) / Y_std; Y_test  = (Y_test_raw - Y_mean) / Y_std

# Clamp velocity/accel before z-score
vel_cols = list(range(N_STEPS+4, N_STEPS+7))
for k in range(N_STEPS):
    base = n_global + k * lag_w
    vel_cols += [base+3, base+4, base+5]
accel_cols = list(range(n_global-3, n_global))
vel_cols += accel_cols
vel_idx_t = torch.tensor(vel_cols)
q_low  = X_train_raw[:, vel_idx_t].quantile(0.01, dim=0)
q_high = X_train_raw[:, vel_idx_t].quantile(0.99, dim=0)
X_train_raw[:, vel_idx_t] = torch.clamp(X_train_raw[:, vel_idx_t], q_low, q_high)
X_test_raw[:,  vel_idx_t] = torch.clamp(X_test_raw[:,  vel_idx_t], q_low, q_high)

X_mean = X_train_raw.mean(0); X_std = X_train_raw.std(0) + 1e-8
X_train = (X_train_raw - X_mean) / X_std; X_test  = (X_test_raw - X_mean) / X_std

for name, t in [('X_train', X_train), ('X_test', X_test),
                ('Y_train', Y_train), ('Y_test',  Y_test)]:
    n_nan = torch.isnan(t).sum().item(); n_inf = torch.isinf(t).sum().item()
    absmax = t.abs().max().item()
    print(f"  {name}: shape={list(t.shape)}  nan={n_nan}  inf={n_inf}  absmax={absmax:.3f}")
    assert n_nan == 0 and n_inf == 0 and absmax < 1e6
print("SANITY PASSED\n")

# ── Model ──────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if ARCH == 'seq_attn':
    model = LiverSeqAttnFlex(N_OUT, N_STEPS, N_NEIGHBOURS, n_nb_feat).to(device)
elif ARCH == 'dual_attn':
    model = LiverDualAttnFlex(N_OUT, N_STEPS, N_NEIGHBOURS, n_nb_feat).to(device)
else:
    raise ValueError(f"Unknown arch: {ARCH}")

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2, eta_min=1e-6)

MODEL_BEST = f'{SOFA_ROOT}/pinn_project/train/membrane_{RUN_ID}_best.pth'
dataset = TensorDataset(X_train.to(device), Y_train.to(device), S_train_raw.to(device))
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
y_std_f  = Y_std[:N_FORCE].detach().clone().to(device)
y_mean_f = Y_mean[:N_FORCE].detach().clone().to(device)

best_val_loss = float('inf'); best_epoch = 0
history = {'epoch': [], 'total': [], 'force': [], 'deform': []}

print("\nTraining...\n")
for epoch in range(N_EPOCHS):
    model.train()
    epoch_total = epoch_force = epoch_deform = 0.0
    for X_b, Y_b, S_b in loader:
        pred   = model(X_b)
        pred_f = pred[:, :N_FORCE] * y_std_f + y_mean_f
        true_f = Y_b[:, :N_FORCE]  * y_std_f + y_mean_f
        f_mag  = torch.norm(true_f, dim=1, keepdim=True).detach()
        f_w    = 1.0 + (f_mag / (f_mag.mean() + 1e-8)) ** HIGH_FORCE_GAMMA
        s_w    = 1.0 + SUSTAIN_BETA * S_b.unsqueeze(1)
        L_force  = (f_w * s_w * ((pred[:, :N_FORCE] - Y_b[:, :N_FORCE])**2).mean(1, keepdim=True)).mean()
        if FORCE_ONLY:
            L_deform = torch.tensor(0.0, device=device)
            L = L_force
        else:
            L_deform = ((pred[:, N_FORCE:] - Y_b[:, N_FORCE:])**2).mean()
            L = W_FORCE * L_force + W_DEFORM * L_deform
        optimizer.zero_grad(); L.backward(); optimizer.step()
        epoch_total += L.item(); epoch_force += L_force.item(); epoch_deform += L_deform.item()
    n_b = len(loader)
    scheduler.step(epoch)

    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test.to(device))
            val_loss = ((val_pred - Y_test.to(device)) ** 2).mean().item()
        model.train()
        if val_loss < best_val_loss:
            best_val_loss = val_loss; best_epoch = epoch
            torch.save(model.state_dict(), MODEL_BEST)
        elif epoch - best_epoch >= EARLY_STOP:
            print(f"Early stopping at epoch {epoch} (best={best_epoch}, val={best_val_loss:.6f})")
            break

    if epoch % 200 == 0:
        history['epoch'].append(epoch); history['total'].append(epoch_total/n_b)
        history['force'].append(epoch_force/n_b); history['deform'].append(epoch_deform/n_b)
        print(f"Epoch {epoch:5d} | total={epoch_total/n_b:.6f} force={epoch_force/n_b:.6f} "
              f"deform={epoch_deform/n_b:.6f} lr={optimizer.param_groups[0]['lr']:.2e} "
              f"val_best={best_val_loss:.6f}")

print(f"\nBest: epoch {best_epoch}, val_loss={best_val_loss:.6f}")
model.load_state_dict(torch.load(MODEL_BEST, map_location=device))

# ── Validation metrics ─────────────────────────────────────────────────────────
def inv_log1p_signed(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

model.eval()
with torch.no_grad():
    pred_norm = model(X_test.to(device)).cpu()
    pred = pred_norm * Y_std + Y_mean
    true = Y_test   * Y_std + Y_mean

fp = inv_log1p_signed(pred[:, :N_FORCE])
ft = inv_log1p_signed(true[:, :N_FORCE])
force_rel_err = torch.norm(fp - ft) / (torch.norm(ft) + 1e-8)
per_err = torch.norm(fp - ft, dim=1) / (torch.norm(ft, dim=1) + 1e-8)
thresh  = per_err.quantile(0.9).item()
mask_r  = per_err < thresh
force_rel_robust = torch.norm(fp[mask_r] - ft[mask_r]) / (torch.norm(ft[mask_r]) + 1e-8)

sust_m = S_test_raw.bool()
force_sust = (torch.norm(fp[sust_m] - ft[sust_m]) / (torch.norm(ft[sust_m]) + 1e-8)
              if sust_m.sum() > 0 else torch.tensor(float('nan')))

if FORCE_ONLY:
    deform_rel = torch.tensor(float('nan'))
    max_mm = mean_mm = float('nan')
else:
    dp = pred[:, N_FORCE:]; dt_out = true[:, N_FORCE:]
    deform_rel = torch.norm(dp - dt_out) / (torch.norm(dt_out) + 1e-8)
    diff = (dp - dt_out).reshape(-1, N_VERTICES, 3)
    per_vert = torch.norm(diff, dim=2)
    max_mm  = per_vert.max().item() * 1000
    mean_mm = per_vert.mean().item() * 1000

print(f"\n=== {RUN_ID} RESULTS ===")
print(f"force_rel_l2={force_rel_err.item()*100:.2f}%  "
      f"force_robust={force_rel_robust.item()*100:.2f}%  "
      f"force_sust={force_sust.item()*100:.2f}%")
print(f"deform_rel_l2={deform_rel.item()*100:.2f}%  "
      f"max_vert_err={max_mm:.3f}mm  mean_vert_err={mean_mm:.4f}mm")
contact_mask = torch.norm(ft, dim=1) > 0.5
rel_contact = per_err[contact_mask].mean().item() if contact_mask.sum() > 0 else float('nan')
print(f"mean_rel_contact={rel_contact*100:.2f}%  RMSE={(((fp-ft)**2).mean().sqrt()).item():.4f}N")

with open(RESULTS_FILE, 'a') as f:
    f.write(
        f"run={RUN_ID:<25s}  arch={ARCH:<10s}  n_steps={N_STEPS:2d}  K={N_NEIGHBOURS:2d}  "
        f"n_nb_feat={n_nb_feat:2d}  "
        f"accstress={int(USE_ACCSTRESS)}  strain={int(USE_STRAIN)}  "
        f"stress={int(USE_STRESS)}  proxy={int(USE_PROXY)}  "
        f"force_only={int(FORCE_ONLY)}  gamma={HIGH_FORCE_GAMMA:.1f}  "
        f"best_epoch={best_epoch:5d}  val_loss={best_val_loss:.6f}  "
        f"force_rel={force_rel_err.item()*100:.2f}%  "
        f"force_robust={force_rel_robust.item()*100:.2f}%  "
        f"force_sust={force_sust.item()*100:.2f}%  "
        f"deform_rel={deform_rel.item()*100:.2f}%  "
        f"max_mm={max_mm:.3f}  mean_mm={mean_mm:.4f}  "
        f"params={n_params:,}\n"
    )
print(f"\nSummary appended to {RESULTS_FILE}")

# Save full checkpoint
torch.save({
    'model_state': model.state_dict(),
    'arch': ARCH, 'run_id': RUN_ID,
    'n_steps': N_STEPS, 'n_neighbours': N_NEIGHBOURS, 'n_nb_feat': n_nb_feat,
    'n_global': n_global, 'lag_w': lag_w, 'n_inputs': N_INPUTS, 'n_output': N_OUT,
    'n_force': N_FORCE, 'n_vertices': N_VERTICES,
    'X_mean': X_mean, 'X_std': X_std, 'Y_mean': Y_mean, 'Y_std': Y_std,
    'vel_idx': vel_idx_t, 'q_low': q_low, 'q_high': q_high,
    'use_accstress': USE_ACCSTRESS, 'use_strain': USE_STRAIN,
    'use_stress': USE_STRESS, 'use_proxy': USE_PROXY,
    'force_rel_err': force_rel_err.item(),
    'deform_rel_err': deform_rel.item(),
}, f'{SOFA_ROOT}/pinn_project/train/membrane_{RUN_ID}.pth')
print(f"Full checkpoint saved to membrane_{RUN_ID}.pth")

# Plot
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(history['epoch'], history['total'],  label='total')
axs[0].plot(history['epoch'], history['force'],  label='force')
axs[0].plot(history['epoch'], history['deform'], label='deform')
axs[0].set_yscale('log'); axs[0].set_xlabel('epoch'); axs[0].legend()
axs[0].set_title(f'{RUN_ID} — membrane training loss')
axs[1].plot(ft[:200, 1].numpy(), label='true fy', alpha=0.7)
axs[1].plot(fp[:200, 1].numpy(), label='pred fy', alpha=0.7)
axs[1].legend(); axs[1].set_title('Force fy — first 200 val samples')
plt.tight_layout()
plt.savefig(f'{SOFA_ROOT}/pinn_project/train/membrane_{RUN_ID}_loss.png', dpi=150)
print("Plot saved.")
