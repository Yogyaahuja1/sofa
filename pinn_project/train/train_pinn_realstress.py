"""
PINN Training Script — Real-Stress Ablation
================================================================================
Same pipeline as train_pinn_contactweight.py (n_lags=8, sustain_beta=5.0 — the
established best config), with ONE addition: a 4th per-lag neighbourhood block,
nb_realstress (rsxx/rsyy/rszz from getLastStress(), the real Hooke's-law stress
that's been sitting unused in training_data.csv). Tests whether it adds signal
beyond the already-used real strain (rexx/reyy/rezz) and stress-proxy
(sax/say/saz) — a proper ablation (train with vs without), not just a
standalone correlation check, since a feature can help in combination even
with low standalone target-correlation.

Run WITHOUT stress (baseline, same as best contactweight run) and WITH stress
(this script) on the same epoch budget, then compare force_rel_l2_sustained.
"""
import os as _os
SOFA_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pinn_model import LagSequenceAttentionAccelStress

parser = argparse.ArgumentParser()
parser.add_argument('--n-lags', type=int, default=8)
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--sustain-beta', type=float, default=5.0)
args = parser.parse_args()

CSV_PATH      = f'{SOFA_ROOT}/pinn_project/data/training_data.csv'
N_VERTICES    = 181
FIXED_INDICES = [3, 39, 64]
BATCH_SIZE    = 64
N_EPOCHS      = args.epochs
LR            = 3e-4
TRAIN_SPLIT   = 0.8
N_NEIGHBOURS  = 20
N_LAGS        = args.n_lags
W_FORCE       = 0.5
W_DEFORM      = 1.0
SEED          = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print("Loading data...")
df = pd.read_csv(CSV_PATH)

required_time_cols = {'sim_time', 'dt_since_last'}
missing_time_cols = required_time_cols - set(df.columns)
if missing_time_cols:
    raise ValueError(f"CSV is missing columns {missing_time_cols}")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

sess_counts = df.groupby('session_id')['session_id'].transform('count')
df = df[sess_counts >= N_LAGS * 3].reset_index(drop=True)
print(f"After short-session filter: {len(df)} rows")

max_disp = df[[c for c in df.columns if c.startswith('dx') or c.startswith('dy') or c.startswith('dz')]].values
pdmax    = df[[c for c in df.columns if c.startswith('pdx') or c.startswith('pdy') or c.startswith('pdz')]].values
delta_check = np.abs(max_disp - pdmax).max(axis=1)
df = df[delta_check < 2.0].reset_index(drop=True)
print(f"Rows after pre-filtering df: {len(df)}")

df['tool_fx'] = df['tool_fx'].clip(-100.0, 100.0)
df['tool_fy'] = df['tool_fy'].clip(-100.0, 100.0)
df['tool_fz'] = df['tool_fz'].clip(-100.0, 100.0)
print(f"Rows after cleaning: {len(df)}")
N_rows = len(df)

df_fmag = np.linalg.norm(df[['tool_fx', 'tool_fy', 'tool_fz']].values, axis=1)
is_sustained = np.zeros(N_rows, dtype=np.float32)
for sid in df['session_id'].unique():
    if sid == 0:
        continue
    idx = df.index[df['session_id'] == sid].values
    high = df_fmag[idx] > 3.0
    run_id = np.zeros(len(high), dtype=int)
    cur = 0
    for i in range(len(high)):
        cur = cur + 1 if high[i] else 0
        run_id[i] = cur
    if run_id.max() >= 10:
        is_sustained[idx[high]] = 1.0
print(f"Sustained-contact rows flagged: {int(is_sustained.sum())} / {N_rows} "
      f"({100*is_sustained.sum()/N_rows:.1f}%)")
sustained_tensor_all = torch.FloatTensor(is_sustained.copy())

dy_cols_diag = [c for c in df.columns if c.startswith('dy')]
dy_max = df[dy_cols_diag].abs().max()
best_vertex_id = None
for col in dy_max.sort_values(ascending=False).index:
    vid = int(col[2:])
    if vid not in FIXED_INDICES:
        best_vertex_id = vid
        break
if best_vertex_id is None:
    best_vertex_id = 0
print(f"Most active vertex: {best_vertex_id}")

vertex_pos = np.load(f'{SOFA_ROOT}/pinn_project/data/liver_vertices.npy')
tool_positions = df[['tool_x', 'tool_y', 'tool_z']].values

print("Computing neighbourhood indices...")
neighbour_idx    = np.zeros((N_rows, N_NEIGHBOURS), dtype=int)
min_dist_to_mesh = np.zeros(N_rows, dtype=np.float32)
for row in range(N_rows):
    dists = np.linalg.norm(vertex_pos - tool_positions[row], axis=1)
    sorted_idx = np.argsort(dists)
    neighbour_idx[row]    = sorted_idx[:N_NEIGHBOURS]
    min_dist_to_mesh[row] = dists[sorted_idx[0]]
print(f"Neighbourhood computed.")

dx_all  = df[[f'dx{i}'  for i in range(181)]].values
dy_all  = df[[f'dy{i}'  for i in range(181)]].values
dz_all  = df[[f'dz{i}'  for i in range(181)]].values
pdx_all = df[[f'pdx{i}' for i in range(181)]].values
pdy_all = df[[f'pdy{i}' for i in range(181)]].values
pdz_all = df[[f'pdz{i}' for i in range(181)]].values

ddx_all = dx_all - pdx_all
ddy_all = dy_all - pdy_all
ddz_all = dz_all - pdz_all
sax_all = df[[f'sax{i}' for i in range(181)]].values
say_all = df[[f'say{i}' for i in range(181)]].values
saz_all = df[[f'saz{i}' for i in range(181)]].values

rexx_all = df[[f'rexx{i}' for i in range(181)]].values
reyy_all = df[[f'reyy{i}' for i in range(181)]].values
rezz_all = df[[f'rezz{i}' for i in range(181)]].values

# NEW: real (Hooke's law) per-vertex STRESS from TetrahedronFEMForceField — the
# feature this ablation tests, never used in any prior model.
rsxx_all = df[[f'rsxx{i}' for i in range(181)]].values
rsyy_all = df[[f'rsyy{i}' for i in range(181)]].values
rszz_all = df[[f'rszz{i}' for i in range(181)]].values

print("Building neighbourhood lag features...")
nb_deform_features     = np.zeros((N_rows, N_NEIGHBOURS * 3 * N_LAGS), dtype=np.float32)
nb_stress_features     = np.zeros((N_rows, N_NEIGHBOURS * 3 * N_LAGS), dtype=np.float32)
nb_realstrain_features = np.zeros((N_rows, N_NEIGHBOURS * 3 * N_LAGS), dtype=np.float32)
nb_realstress_features = np.zeros((N_rows, N_NEIGHBOURS * 3 * N_LAGS), dtype=np.float32)
session_ids = df['session_id'].values

for row in range(N_rows):
    nb = neighbour_idx[row]
    deform_row, stress_row, realstrain_row, realstress_row = [], [], [], []
    for lag in range(1, N_LAGS + 1):
        src_row = max(0, row - lag)
        if session_ids[src_row] != session_ids[row]:
            deform_row     += [np.zeros(N_NEIGHBOURS)] * 3
            stress_row     += [np.zeros(N_NEIGHBOURS)] * 3
            realstrain_row += [np.zeros(N_NEIGHBOURS)] * 3
            realstress_row += [np.zeros(N_NEIGHBOURS)] * 3
        else:
            deform_row.append(ddx_all[src_row][nb]); deform_row.append(ddy_all[src_row][nb]); deform_row.append(ddz_all[src_row][nb])
            stress_row.append(sax_all[src_row][nb]); stress_row.append(say_all[src_row][nb]); stress_row.append(saz_all[src_row][nb])
            realstrain_row.append(rexx_all[src_row][nb]); realstrain_row.append(reyy_all[src_row][nb]); realstrain_row.append(rezz_all[src_row][nb])
            realstress_row.append(rsxx_all[src_row][nb]); realstress_row.append(rsyy_all[src_row][nb]); realstress_row.append(rszz_all[src_row][nb])
    nb_deform_features[row]     = np.concatenate(deform_row)
    nb_stress_features[row]     = np.concatenate(stress_row)
    nb_realstrain_features[row] = np.concatenate(realstrain_row)
    nb_realstress_features[row] = np.concatenate(realstress_row)

print(f"Neighbourhood deform features shape:      {nb_deform_features.shape}")
print(f"Neighbourhood stress features shape:      {nb_stress_features.shape}")
print(f"Neighbourhood real-strain features shape: {nb_realstrain_features.shape}")
print(f"Neighbourhood real-stress features shape: {nb_realstress_features.shape}")

from scipy.stats import pearsonr
delta_best = dy_all[:, best_vertex_id] - pdy_all[:, best_vertex_id]
corr_realstress, _ = pearsonr(nb_realstress_features[:, 0], delta_best)
print(f"Neighbour real-stress lag1 corr with target delta:     {corr_realstress:.3f}")

tool_x_all  = df['tool_x'].values
tool_y_all  = df['tool_y'].values
tool_z_all  = df['tool_z'].values
tool_vx_all = df['tool_vx'].values
tool_vy_all = df['tool_vy'].values
tool_vz_all = df['tool_vz'].values
tool_fx_all = df['tool_fx'].values
tool_fy_all = df['tool_fy'].values
tool_fz_all = df['tool_fz'].values
real_time_vals = df['real_time'].values

tool_hist_features = np.zeros((N_rows, 9 * N_LAGS), dtype=np.float32)
run_start_idx = np.zeros(N_rows, dtype=np.int64)
for row in range(1, N_rows):
    run_start_idx[row] = row if session_ids[row] != session_ids[row - 1] else run_start_idx[row - 1]

for row in range(N_rows):
    lag_vals = []
    sess_start = run_start_idx[row]
    for lag in range(1, N_LAGS + 1):
        src_row = max(0, row - lag)
        if session_ids[src_row] != session_ids[row]:
            lag_vals.append([
                tool_x_all[sess_start], tool_y_all[sess_start], tool_z_all[sess_start],
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ])
        else:
            lag_vals.append([
                tool_x_all[src_row],  tool_y_all[src_row],  tool_z_all[src_row],
                tool_vx_all[src_row], tool_vy_all[src_row], tool_vz_all[src_row],
                np.sign(tool_fx_all[src_row]) * np.log1p(np.abs(tool_fx_all[src_row])),
                np.sign(tool_fy_all[src_row]) * np.log1p(np.abs(tool_fy_all[src_row])),
                np.sign(tool_fz_all[src_row]) * np.log1p(np.abs(tool_fz_all[src_row])),
            ])
    tool_hist_features[row] = np.concatenate(lag_vals)
print(f"Tool history features shape: {tool_hist_features.shape}")

dt_pred_vals = np.zeros((N_rows, 1), dtype=np.float32)
for row in range(N_rows):
    base_row = max(0, row - N_LAGS)
    dt_pred_vals[row, 0] = real_time_vals[row] - real_time_vals[base_row]
dt_pred_vals = np.clip(dt_pred_vals, 0.0, 0.15)
current_pos_vel = np.stack([
    tool_x_all, tool_y_all, tool_z_all, tool_vx_all, tool_vy_all, tool_vz_all,
], axis=1).astype(np.float32)

delta_min_dist = np.zeros(N_rows, dtype=np.float32)
delta_min_dist[1:] = min_dist_to_mesh[1:] - min_dist_to_mesh[:-1]

force_mag_all = np.linalg.norm(df[['tool_fx', 'tool_fy', 'tool_fz']].values, axis=1).astype(np.float32)
lag1_contact = np.zeros(N_rows, dtype=np.float32)
lag1_contact[1:] = (force_mag_all[:-1] > 0.01).astype(np.float32)

contact_features = np.stack([min_dist_to_mesh, delta_min_dist, lag1_contact], axis=1).astype(np.float32)

accel = np.zeros((N_rows, 3), dtype=np.float32)
for row in range(N_rows):
    src1, src3 = max(0, row - 1), max(0, row - 3)
    if session_ids[src1] != session_ids[row] or session_ids[src3] != session_ids[row]:
        continue
    dt = real_time_vals[src1] - real_time_vals[src3]
    if dt < 1e-4:
        continue
    accel[row, 0] = (tool_vx_all[src1] - tool_vx_all[src3]) / dt
    accel[row, 1] = (tool_vy_all[src1] - tool_vy_all[src3]) / dt
    accel[row, 2] = (tool_vz_all[src1] - tool_vz_all[src3]) / dt
accel = np.clip(accel, -500.0, 500.0)

dt_cum = np.zeros((N_rows, N_LAGS), dtype=np.float32)
for row in range(N_rows):
    base_row = max(0, row - N_LAGS)
    rt_base = real_time_vals[base_row]
    for lag in range(1, N_LAGS + 1):
        src_row = max(0, row - lag)
        if session_ids[src_row] != session_ids[row]:
            dt_cum[row, lag - 1] = 0.0
        else:
            dt_cum[row, lag - 1] = real_time_vals[src_row] - rt_base
dt_cum = np.clip(dt_cum, 0.0, 0.15)

X_combined = np.concatenate([
    dt_cum, dt_pred_vals, current_pos_vel, contact_features,
    tool_hist_features, nb_deform_features, nb_stress_features,
    nb_realstrain_features, nb_realstress_features,  # NEW block
    accel,
], axis=1).astype(np.float32)

N_INPUTS = X_combined.shape[1]
X_raw_tensor = torch.FloatTensor(X_combined.copy())
print(f"Total inputs: {N_INPUTS}  (expected {250*N_LAGS+13})")

n_global_const = N_LAGS + 10
vel_indices = [9, 10, 11]
for k in range(N_LAGS):
    base = n_global_const + 9 * k
    vel_indices += [base + 3, base + 4, base + 5]
accel_off = n_global_const + 9*N_LAGS + 60*N_LAGS*3
vel_indices += [accel_off, accel_off + 1, accel_off + 2]

active_verts = [i for i in range(N_VERTICES) if i not in FIXED_INDICES]
N_OUT_DEFORM = len(active_verts) * 3

dy_cols, pdy_cols = [], []
for i in active_verts:
    dy_cols  += [f'dx{i}',  f'dy{i}',  f'dz{i}']
    pdy_cols += [f'pdx{i}', f'pdy{i}', f'pdz{i}']

Y_deform = (df[dy_cols].values - df[pdy_cols].values).astype(np.float32)

force_raw = df[['tool_fx','tool_fy','tool_fz']].values.astype(np.float32)
Y_force = np.sign(force_raw) * np.log1p(np.abs(force_raw))

N_FORCE = 3
Y_combined = np.concatenate([Y_force, Y_deform], axis=1).astype(np.float32)
N_OUT = Y_combined.shape[1]
Y_raw_tensor = torch.FloatTensor(Y_combined.copy())
print(f"Total outputs: {N_OUT} (force={N_FORCE}, deform={N_OUT_DEFORM})")

perm_all = torch.randperm(N_rows)
X_all = X_raw_tensor[perm_all]
Y_all = Y_raw_tensor[perm_all]
sustained_all = sustained_tensor_all[perm_all]

n_train = int(TRAIN_SPLIT * len(X_all))
X_train_raw = X_all[:n_train]
Y_train_raw = Y_all[:n_train]
X_test_raw  = X_all[n_train:]
sustained_test_raw = sustained_all[n_train:]
Y_test_raw  = Y_all[n_train:]
sustained_train_raw = sustained_all[:n_train]

perm = torch.randperm(len(X_train_raw))
X_train_raw = X_train_raw[perm]
Y_train_raw = Y_train_raw[perm]
sustained_train_raw = sustained_train_raw[perm]

print(f"  Train: {len(X_train_raw)}  |  Test: {len(X_test_raw)}")

Y_mean = Y_train_raw.mean(dim=0)
Y_std  = Y_train_raw.std(dim=0) + 1e-8
Y_train = (Y_train_raw - Y_mean) / Y_std
Y_test  = (Y_test_raw  - Y_mean) / Y_std

vel_idx_t = torch.tensor(vel_indices)
vel_q_low  = X_train_raw[:, vel_idx_t].quantile(0.01, dim=0)
vel_q_high = X_train_raw[:, vel_idx_t].quantile(0.99, dim=0)
X_train_raw[:, vel_idx_t] = torch.clamp(X_train_raw[:, vel_idx_t], vel_q_low, vel_q_high)
X_test_raw[:,  vel_idx_t] = torch.clamp(X_test_raw[:,  vel_idx_t], vel_q_low, vel_q_high)

X_mean = X_train_raw.mean(dim=0)
X_std  = X_train_raw.std(dim=0) + 1e-8
X_train = (X_train_raw - X_mean) / X_std
X_test  = (X_test_raw  - X_mean) / X_std

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

model = LagSequenceAttentionAccelStress(n_output=N_OUT, n_inputs=N_INPUTS, n_lags=N_LAGS).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=1000, T_mult=2, eta_min=1e-6
)

SUSTAIN_BETA = args.sustain_beta

dataset  = TensorDataset(X_train.to(device), Y_train.to(device), sustained_train_raw.to(device))
loader   = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
y_std_f  = Y_std[:N_FORCE].detach().clone().to(device)
y_mean_f = Y_mean[:N_FORCE].detach().clone().to(device)

best_val_loss = float('inf')
best_epoch    = 0
MODEL_BEST    = f'tissue_pinn_realstress_n{N_LAGS}_best.pth'
EARLY_STOP_PATIENCE = 800

print("\nTraining...\n")
for epoch in range(N_EPOCHS):
    model.train()
    epoch_total = epoch_force = epoch_deform = 0.0

    for X_batch, Y_batch, sustain_batch in loader:
        u_pred = model(X_batch)

        pred_f_N = u_pred[:, :N_FORCE] * y_std_f + y_mean_f
        true_f_N = Y_batch[:, :N_FORCE] * y_std_f + y_mean_f
        f_mag    = torch.norm(true_f_N, dim=1, keepdim=True).detach()
        f_weight = 1.0 + f_mag / (f_mag.mean() + 1e-8)
        contact_weight = 1.0 + SUSTAIN_BETA * sustain_batch.unsqueeze(1)
        L_force  = (f_weight * contact_weight * ((u_pred[:, :N_FORCE] - Y_batch[:, :N_FORCE]) ** 2).mean(dim=1, keepdim=True)).mean()
        L_deform = torch.mean((u_pred[:, N_FORCE:] - Y_batch[:, N_FORCE:]) ** 2)
        L = W_FORCE * L_force + W_DEFORM * L_deform

        optimizer.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_total  += L.item()
        epoch_force  += L_force.item()
        epoch_deform += L_deform.item()

    n_batches = len(loader)
    scheduler.step(epoch)

    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test.to(device))
            val_loss = torch.mean((val_pred - Y_test.to(device)) ** 2).item()
        model.train()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save(model.state_dict(), MODEL_BEST)
        elif epoch - best_epoch >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (best epoch {best_epoch}, val loss {best_val_loss:.6f})")
            break

    if epoch % 200 == 0:
        print(f"Epoch {epoch:5d} | Total: {epoch_total/n_batches:.6f} | "
              f"Force: {epoch_force/n_batches:.6f} | Deform: {epoch_deform/n_batches:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

print(f"\nBest checkpoint: epoch {best_epoch} (val loss {best_val_loss:.6f}) -> {MODEL_BEST}")
model.load_state_dict(torch.load(MODEL_BEST, map_location=device))

def inv_log1p_signed(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

print("\n=== Validation vs FEM ===")
model.eval()
with torch.no_grad():
    u_pred_norm = model(X_test.to(device)).cpu()
    u_pred = (u_pred_norm * Y_std) + Y_mean
    u_true = (Y_test * Y_std) + Y_mean

    force_pred_log = u_pred[:, :N_FORCE]
    force_true_log = u_true[:, :N_FORCE]
    force_pred_N = inv_log1p_signed(force_pred_log)
    force_true_N = inv_log1p_signed(force_true_log)

    force_rel_err = torch.norm(force_pred_N - force_true_N) / (torch.norm(force_true_N) + 1e-8)
    per_sample_force_err = torch.norm(force_pred_N - force_true_N, dim=1)

    n_test = len(per_sample_force_err)
    k_outliers = max(1, n_test // 100)
    err_thresh = torch.topk(per_sample_force_err, k_outliers).values.min()
    robust_mask = per_sample_force_err < err_thresh
    force_rel_err_robust = torch.norm(force_pred_N[robust_mask] - force_true_N[robust_mask]) / \
                           (torch.norm(force_true_N[robust_mask]) + 1e-8)

    sustain_mask = sustained_test_raw.bool()
    if sustain_mask.sum() > 0:
        force_rel_err_sustained = torch.norm(force_pred_N[sustain_mask] - force_true_N[sustain_mask]) / \
                                   (torch.norm(force_true_N[sustain_mask]) + 1e-8)
    else:
        force_rel_err_sustained = torch.tensor(float('nan'))
    print(f"Sustained-contact test rows: {int(sustain_mask.sum())} / {n_test}")
    print(f"Relative L2 error (force, sustained-contact rows only): {force_rel_err_sustained.item()*100:.2f}%")

    print(f"\n--- FORCE (primary output) ---")
    print(f"Relative L2 error (force):  {force_rel_err.item()*100:.2f}%")
    print(f"Relative L2 (excl. worst {k_outliers}/{n_test} samples): {force_rel_err_robust.item()*100:.2f}%")

    deform_pred = u_pred[:, N_FORCE:]
    deform_true = u_true[:, N_FORCE:]
    deform_rel_err = torch.norm(deform_pred - deform_true) / (torch.norm(deform_true) + 1e-8)
    print(f"\n--- DEFORMATION (secondary output) ---")
    print(f"Relative L2 error (deform): {deform_rel_err.item()*100:.2f}%")

torch.save({
    'model_state': model.state_dict(),
    'n_output':    N_OUT,
    'n_inputs':    N_INPUTS,
    'n_force':     N_FORCE,
    'n_vertices':  N_VERTICES,
    'X_mean':      X_mean,
    'X_std':       X_std,
    'Y_mean':      Y_mean,
    'Y_std':       Y_std,
    'vel_indices': vel_indices,
    'vel_q_low':   vel_q_low,
    'vel_q_high':  vel_q_high,
    'force_log_transform': True,
    'epoch':       N_EPOCHS,
    'force_rel_err': force_rel_err.item(),
    'force_rel_err_robust': force_rel_err_robust.item(),
    'force_rel_err_sustained': force_rel_err_sustained.item(),
    'deform_rel_err': deform_rel_err.item(),
    'n_lags': N_LAGS,
    'sustain_beta': SUSTAIN_BETA,
}, f'tissue_pinn_realstress_n{N_LAGS}_beta{SUSTAIN_BETA}.pth')
print(f"Model saved to tissue_pinn_realstress_n{N_LAGS}_beta{SUSTAIN_BETA}.pth")

with open('realstress_results.txt', 'a') as f:
    f.write(f"n_lags={N_LAGS:3d}  sustain_beta={SUSTAIN_BETA:.2f}  WITH_REALSTRESS  epochs_run={best_epoch:5d}  "
            f"best_val_loss={best_val_loss:.6f}  "
            f"force_rel_l2={force_rel_err.item()*100:6.2f}%  "
            f"force_rel_l2_robust={force_rel_err_robust.item()*100:6.2f}%  "
            f"force_rel_l2_sustained={force_rel_err_sustained.item()*100:6.2f}%  "
            f"deform_rel_l2={deform_rel_err.item()*100:6.2f}%  "
            f"params={n_params:,}\n")
print("Summary appended to realstress_results.txt")
