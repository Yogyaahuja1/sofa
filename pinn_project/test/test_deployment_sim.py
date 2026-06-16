"""
Deployment Simulation Test — FEM Latency Scenarios
====================================================
Simulates real deployment where FEM (SOFA) is NOT always available
before the haptic device needs a force update.

Logic per step:
  - Tool kinematics (pos/vel/force): always real (haptic sensor)
  - Contact features (min_dist):     always real (tool pos + rest mesh)
  - When FEM available:              use real deform/stress/strain in buffer
  - When FEM NOT available:          use PINN's predicted deform + last known stress/strain

FEM_SKIP = N means FEM result arrives every N haptic steps.
  FEM_SKIP=1  → FEM always available (ideal, matches offline test)
  FEM_SKIP=5  → FEM at 20% of haptic rate
  FEM_SKIP=10 → FEM at 10% of haptic rate
  FEM_SKIP=inf→ FEM never available (full autoregressive, worst case)

Run: python3 test_deployment_sim.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'train'))

import torch
import numpy as np
import pandas as pd
from pinn_model import LiverUNet

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH    = '/home/yogyaahuja/sofa/pinn_project/train/tissue_pinn_force_final.pth'
CSV_PATH      = '/home/yogyaahuja/sofa/pinn_project/data/training_data.csv'
VERTICES_PATH = '/home/yogyaahuja/sofa/pinn_project/data/liver_vertices.npy'
N_NEIGHBOURS  = 20
N_LAGS        = 5
N_VERTICES    = 181
FIXED_INDICES = [3, 39, 64]
SEED          = 42
# FEM available every N steps (9999 = never)
FEM_SKIPS     = [1, 2, 5, 10, 20, 9999]

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("Loading model...")
ckpt   = torch.load(MODEL_PATH, map_location=device, weights_only=False)
X_mean = ckpt['X_mean']
X_std  = ckpt['X_std']
Y_mean = ckpt['Y_mean']
Y_std  = ckpt['Y_std']
n_in   = ckpt['n_inputs']
n_out  = ckpt['n_output']
n_force= ckpt['n_force']

model = LiverUNet(n_output=n_out, n_inputs=n_in).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"Model loaded: {n_in} inputs → {n_out} outputs ({n_force} force + {n_out-n_force} deform)")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(CSV_PATH).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
N_rows = len(df)
print(f"Rows: {N_rows}")

# Reproduce EXACT same train/test split as training (SEED=42)
rng = np.random.RandomState(SEED)
idx = np.arange(N_rows)
rng.shuffle(idx)
split    = int(0.8 * N_rows)
test_set = set(idx[split:])
print(f"Test rows: {len(test_set)}")

# ── RAW ARRAYS ────────────────────────────────────────────────────────────────
active_verts = [i for i in range(N_VERTICES) if i not in FIXED_INDICES]
FIXED_SET    = set(FIXED_INDICES)

dx_all  = df[[f'dx{i}'   for i in range(N_VERTICES)]].values.astype(np.float32)
dy_all  = df[[f'dy{i}'   for i in range(N_VERTICES)]].values.astype(np.float32)
dz_all  = df[[f'dz{i}'   for i in range(N_VERTICES)]].values.astype(np.float32)
pdx_all = df[[f'pdx{i}'  for i in range(N_VERTICES)]].values.astype(np.float32)
pdy_all = df[[f'pdy{i}'  for i in range(N_VERTICES)]].values.astype(np.float32)
pdz_all = df[[f'pdz{i}'  for i in range(N_VERTICES)]].values.astype(np.float32)
ddx_all = (dx_all - pdx_all)
ddy_all = (dy_all - pdy_all)
ddz_all = (dz_all - pdz_all)

sax_all = df[[f'sax{i}'  for i in range(N_VERTICES)]].values.astype(np.float32)
say_all = df[[f'say{i}'  for i in range(N_VERTICES)]].values.astype(np.float32)
saz_all = df[[f'saz{i}'  for i in range(N_VERTICES)]].values.astype(np.float32)
rxx_all = df[[f'rexx{i}' for i in range(N_VERTICES)]].values.astype(np.float32)
ryy_all = df[[f'reyy{i}' for i in range(N_VERTICES)]].values.astype(np.float32)
rzz_all = df[[f'rezz{i}' for i in range(N_VERTICES)]].values.astype(np.float32)

tool_x  = df['tool_x'].values.astype(np.float32)
tool_y  = df['tool_y'].values.astype(np.float32)
tool_z  = df['tool_z'].values.astype(np.float32)
tool_vx = df['tool_vx'].values.astype(np.float32)
tool_vy = df['tool_vy'].values.astype(np.float32)
tool_vz = df['tool_vz'].values.astype(np.float32)
tool_fx = df['tool_fx'].values.astype(np.float32)
tool_fy = df['tool_fy'].values.astype(np.float32)
tool_fz = df['tool_fz'].values.astype(np.float32)
rt_vals = df['real_time'].values.astype(np.float32)
sess    = df['session_id'].values

# ── NEIGHBOUR INDICES ─────────────────────────────────────────────────────────
print("Computing neighbour indices...")
vertex_pos   = np.load(VERTICES_PATH)
tool_pos_all = np.stack([tool_x, tool_y, tool_z], axis=1)
nb_idx       = np.zeros((N_rows, N_NEIGHBOURS), dtype=int)
min_dist_arr = np.zeros(N_rows, dtype=np.float32)
for r in range(N_rows):
    d = np.linalg.norm(vertex_pos - tool_pos_all[r], axis=1)
    s = np.argsort(d)
    nb_idx[r]       = s[:N_NEIGHBOURS]
    min_dist_arr[r] = d[s[0]]

force_true = np.stack([tool_fx, tool_fy, tool_fz], axis=1)  # (N, 3)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def decode_pred_deform(pred_deform_534):
    """
    Decode model's deform output (534,) → per-vertex arrays (181,) each for dx, dy, dz.
    Output ordering in Y: for each active vertex k: [ddx_k, ddy_k, ddz_k] (interleaved).
    """
    out_x = np.zeros(N_VERTICES, dtype=np.float32)
    out_y = np.zeros(N_VERTICES, dtype=np.float32)
    out_z = np.zeros(N_VERTICES, dtype=np.float32)
    for k, vi in enumerate(active_verts):
        out_x[vi] = pred_deform_534[k * 3 + 0]
        out_y[vi] = pred_deform_534[k * 3 + 1]
        out_z[vi] = pred_deform_534[k * 3 + 2]
    return out_x, out_y, out_z


def build_X(row, buf_ddx, buf_ddy, buf_ddz,
            buf_sax, buf_say, buf_saz,
            buf_rxx, buf_ryy, buf_rzz):
    """
    Build 960-dim feature vector for one row using the provided buffers.
    buf_*: list of N_LAGS arrays (shape 181,), index 0=lag1, index 4=lag5.
    """
    nb = nb_idx[row]

    # dt_cum (5)
    base_r  = max(0, row - N_LAGS)
    rt_base = rt_vals[base_r]
    dt_cum_r = np.zeros(N_LAGS, dtype=np.float32)
    for lag in range(1, N_LAGS + 1):
        src = max(0, row - lag)
        if sess[src] != sess[row]:
            dt_cum_r[lag - 1] = 0.0
        else:
            dt_cum_r[lag - 1] = float(np.clip(rt_vals[src] - rt_base, 0, 0.15))

    # dt_pred (1)
    dt_pred = float(np.clip(rt_vals[row] - rt_base, 0, 0.15))

    # current_pos_vel (6) — always real sensor
    cur = np.array([tool_x[row], tool_y[row], tool_z[row],
                    tool_vx[row], tool_vy[row], tool_vz[row]], dtype=np.float32)

    # contact features (3) — always real (tool pos + rest mesh)
    d_dist = 0.0
    if row > 0 and sess[row - 1] == sess[row]:
        d_dist = float(min_dist_arr[row] - min_dist_arr[row - 1])
    prev_f = float(np.linalg.norm([tool_fx[row-1], tool_fy[row-1], tool_fz[row-1]])) if row > 0 else 0.0
    cf = np.array([min_dist_arr[row], d_dist, float(prev_f > 0.01)], dtype=np.float32)

    # tool_hist (45) — always real sensor
    th = []
    for lag in range(1, N_LAGS + 1):
        src = max(0, row - lag)
        if sess[src] != sess[row]:
            ss = int(np.searchsorted(sess, sess[row]))
            th.extend([tool_x[ss], tool_y[ss], tool_z[ss], 0, 0, 0, 0, 0, 0])
        else:
            th.extend([
                tool_x[src], tool_y[src], tool_z[src],
                tool_vx[src], tool_vy[src], tool_vz[src],
                float(np.sign(tool_fx[src]) * np.log1p(abs(tool_fx[src]))),
                float(np.sign(tool_fy[src]) * np.log1p(abs(tool_fy[src]))),
                float(np.sign(tool_fz[src]) * np.log1p(abs(tool_fz[src]))),
            ])
    th = np.array(th, dtype=np.float32)

    # nb_deform/stress/strain (300 each) — from buffer (real or predicted)
    df_f, sf_f, rf_f = [], [], []
    for lag in range(1, N_LAGS + 1):
        li = lag - 1  # buffer index: lag1=0, lag5=4
        df_f.extend([buf_ddx[li][nb], buf_ddy[li][nb], buf_ddz[li][nb]])
        sf_f.extend([buf_sax[li][nb], buf_say[li][nb], buf_saz[li][nb]])
        rf_f.extend([buf_rxx[li][nb], buf_ryy[li][nb], buf_rzz[li][nb]])

    X = np.concatenate([
        dt_cum_r, [dt_pred], cur, cf, th,
        np.concatenate(df_f),
        np.concatenate(sf_f),
        np.concatenate(rf_f),
    ])
    return X.astype(np.float32)


# ── SIMULATION ────────────────────────────────────────────────────────────────
def run_scenario(fem_skip):
    # Init buffers with real FEM at row 0
    buf_ddx = [ddx_all[0].copy()] * N_LAGS
    buf_ddy = [ddy_all[0].copy()] * N_LAGS
    buf_ddz = [ddz_all[0].copy()] * N_LAGS
    buf_sax = [sax_all[0].copy()] * N_LAGS
    buf_say = [say_all[0].copy()] * N_LAGS
    buf_saz = [saz_all[0].copy()] * N_LAGS
    buf_rxx = [rxx_all[0].copy()] * N_LAGS
    buf_ryy = [ryy_all[0].copy()] * N_LAGS
    buf_rzz = [rzz_all[0].copy()] * N_LAGS

    # Last known stress/strain — used when FEM unavailable
    last_sax = sax_all[0].copy()
    last_say = say_all[0].copy()
    last_saz = saz_all[0].copy()
    last_rxx = rxx_all[0].copy()
    last_ryy = ryy_all[0].copy()
    last_rzz = rzz_all[0].copy()

    errors    = []
    true_mags = []

    for row in range(N_rows):
        # Build feature vector from buffer
        X      = build_X(row, buf_ddx, buf_ddy, buf_ddz,
                         buf_sax, buf_say, buf_saz,
                         buf_rxx, buf_ryy, buf_rzz)
        X_norm = (X - X_mean) / (X_std + 1e-8)
        X_t    = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            Y_norm = model(X_t).cpu().numpy()[0]
        Y_pred = Y_norm * Y_std + Y_mean

        # Inverse log1p on force output
        force_pred_log = Y_pred[:n_force]
        force_pred = np.sign(force_pred_log) * (np.expm1(np.abs(force_pred_log)))

        # Record error for test rows only
        if row in test_set:
            err = np.linalg.norm(force_pred - force_true[row])
            errors.append(err)
            true_mags.append(np.linalg.norm(force_true[row]))

        # ── Update buffer for next step ──────────────────────────────────────
        fem_avail = (row % fem_skip == 0)

        if fem_avail:
            # Real FEM result arrived — use ground truth
            new_ddx = ddx_all[row].copy()
            new_ddy = ddy_all[row].copy()
            new_ddz = ddz_all[row].copy()
            new_sax = sax_all[row].copy(); last_sax = new_sax
            new_say = say_all[row].copy(); last_say = new_say
            new_saz = saz_all[row].copy(); last_saz = new_saz
            new_rxx = rxx_all[row].copy(); last_rxx = new_rxx
            new_ryy = ryy_all[row].copy(); last_ryy = new_ryy
            new_rzz = rzz_all[row].copy(); last_rzz = new_rzz
        else:
            # FEM still computing — use PINN's predicted deform
            pred_deform_534 = Y_pred[n_force:]
            new_ddx, new_ddy, new_ddz = decode_pred_deform(pred_deform_534)
            # Stress/strain: repeat last known real FEM value
            new_sax = last_sax
            new_say = last_say
            new_saz = last_saz
            new_rxx = last_rxx
            new_ryy = last_ryy
            new_rzz = last_rzz

        # Shift buffer forward: lag5←lag4←lag3←lag2←lag1←new
        buf_ddx = [new_ddx] + buf_ddx[:-1]
        buf_ddy = [new_ddy] + buf_ddy[:-1]
        buf_ddz = [new_ddz] + buf_ddz[:-1]
        buf_sax = [new_sax] + buf_sax[:-1]
        buf_say = [new_say] + buf_say[:-1]
        buf_saz = [new_saz] + buf_saz[:-1]
        buf_rxx = [new_rxx] + buf_rxx[:-1]
        buf_ryy = [new_ryy] + buf_ryy[:-1]
        buf_rzz = [new_rzz] + buf_rzz[:-1]

    errors    = np.array(errors)
    true_mags = np.array(true_mags)
    rel_l2 = 100 * np.sqrt(np.sum(errors**2)) / (np.sqrt(np.sum(true_mags**2)) + 1e-8)
    mae    = float(np.mean(errors))
    median = float(np.median(errors))
    # Robust: exclude worst 1%
    k = max(1, len(errors) // 100)
    robust_idx = np.argsort(errors)[:-k]
    rel_l2_robust = 100 * np.sqrt(np.sum(errors[robust_idx]**2)) / (
        np.sqrt(np.sum(true_mags[robust_idx]**2)) + 1e-8)
    return rel_l2, rel_l2_robust, mae, median


# ── RUN ALL SCENARIOS ─────────────────────────────────────────────────────────
print(f"\n{'FEM update rate':>16} | {'Rel L2':>8} | {'Robust L2':>10} | {'MAE':>8} | {'Median':>8}")
print("-" * 65)
for fs in FEM_SKIPS:
    lbl = f"every {fs} steps" if fs < 9999 else "never (PINN only)"
    rel_l2, robust, mae, med = run_scenario(fs)
    print(f"{lbl:>16} | {rel_l2:>7.2f}% | {robust:>9.2f}% | {mae:>7.4f}N | {med:>7.4f}N")

print("\nDone.")
print("FEM_SKIP=1 should match offline test result (~33%).")
print("Watch how error grows as FEM becomes less frequent.")
