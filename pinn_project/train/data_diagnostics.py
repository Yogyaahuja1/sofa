"""
Quick pre-training diagnostics — runs the same data prep/normalisation
pipeline as train_pinn.py (no actual training) and reports:
  - NaN/Inf / row-count sanity
  - dt_since_last / sim_time variance after filtering
  - zero-std input columns after normalisation
  - correlation of key feature groups with the prediction target
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

CSV_PATH      = '/home/yogyaahuja/sofa/pinn_project/data/training_data.csv'
N_VERTICES    = 181
FIXED_INDICES = [3, 39, 64]
TRAIN_SPLIT   = 0.8

print("Loading data...")
df = pd.read_csv(CSV_PATH)
print(f"Raw rows: {len(df)}, cols: {len(df.columns)}")

required_time_cols = {'sim_time', 'dt_since_last'}
missing_time_cols = required_time_cols - set(df.columns)
assert not missing_time_cols, f"Missing {missing_time_cols}"

n_nan_inf = df.isin([np.inf, -np.inf]).sum().sum() + df.isna().sum().sum()
print(f"NaN/Inf cells in raw CSV: {n_nan_inf}")

df = df.replace([np.inf, -np.inf], np.nan).dropna()
print(f"Rows after dropna: {len(df)}")

force_mag_raw = np.linalg.norm(df[['tool_fx','tool_fy','tool_fz']].values, axis=1)
df = df[force_mag_raw > 0.01].reset_index(drop=True)
print(f"Rows after force>0.01 filter: {len(df)}")

max_disp = df[[c for c in df.columns if c.startswith('dx') or c.startswith('dy') or c.startswith('dz')]].values
pdmax    = df[[c for c in df.columns if c.startswith('pdx') or c.startswith('pdy') or c.startswith('pdz')]].values
delta_check = np.abs(max_disp - pdmax).max(axis=1)
df = df[delta_check < 2.0].reset_index(drop=True)
print(f"Rows after delta<2.0 filter: {len(df)}")

# ── Time columns sanity ────────────────────────────────────────
print("\n--- sim_time / dt_since_last ---")
print("sim_time   min/max/mean/std:", df['sim_time'].min(), df['sim_time'].max(), df['sim_time'].mean(), df['sim_time'].std())
print("dt_since_last unique values:", df['dt_since_last'].unique()[:10], "... n_unique=", df['dt_since_last'].nunique())
print("dt_since_last min/max/mean/std:", df['dt_since_last'].min(), df['dt_since_last'].max(), df['dt_since_last'].mean(), df['dt_since_last'].std())

# Force clip
df['tool_fx'] = df['tool_fx'].clip(-100.0, 100.0)
df['tool_fy'] = df['tool_fy'].clip(-100.0, 100.0)
df['tool_fz'] = df['tool_fz'].clip(-100.0, 100.0)

base_cols = ['tool_x','tool_y','tool_z','tool_vx','tool_vy','tool_vz',
              'tool_fx','tool_fy','tool_fz','sim_time','dt_since_last']

vertex_pos = np.load('/home/yogyaahuja/sofa/pinn_project/data/liver_vertices.npy')
tool_positions = df[['tool_x','tool_y','tool_z']].values

N_NEIGHBOURS = 20
N_LAGS = 5
neighbour_idx = np.zeros((len(df), N_NEIGHBOURS), dtype=int)
for row in range(len(df)):
    dists = np.linalg.norm(vertex_pos - tool_positions[row], axis=1)
    neighbour_idx[row] = np.argsort(dists)[:N_NEIGHBOURS]

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

N_rows = len(df)
nb_deform_features = np.zeros((N_rows, N_NEIGHBOURS * 3 * N_LAGS))
nb_stress_features = np.zeros((N_rows, N_NEIGHBOURS * 3 * N_LAGS))
for row in range(N_rows):
    nb = neighbour_idx[row]
    deform_row, stress_row = [], []
    for lag in range(1, N_LAGS + 1):
        src_row = max(0, row - lag)
        deform_row += [ddx_all[src_row][nb], ddy_all[src_row][nb], ddz_all[src_row][nb]]
        stress_row += [sax_all[src_row][nb], say_all[src_row][nb], saz_all[src_row][nb]]
    nb_deform_features[row] = np.concatenate(deform_row)
    nb_stress_features[row] = np.concatenate(stress_row)

X_base = df[base_cols].values.copy()
X_base[:, 6] = np.sign(X_base[:, 6]) * np.log1p(np.abs(X_base[:, 6]))
X_base[:, 7] = np.sign(X_base[:, 7]) * np.log1p(np.abs(X_base[:, 7]))
X_base[:, 8] = np.sign(X_base[:, 8]) * np.log1p(np.abs(X_base[:, 8]))

X_combined = np.concatenate([X_base, nb_deform_features, nb_stress_features], axis=1)
N_INPUTS = X_combined.shape[1]
print(f"\nTotal inputs: {N_INPUTS}")

active_verts = [i for i in range(N_VERTICES) if i not in FIXED_INDICES]
dy_cols, pdy_cols = [], []
for i in active_verts:
    dy_cols  += [f'dx{i}', f'dy{i}', f'dz{i}']
    pdy_cols += [f'pdx{i}', f'pdy{i}', f'pdz{i}']
Y_delta = (df[dy_cols].values - df[pdy_cols].values).astype(np.float32)
print(f"Output (Y_delta) shape: {Y_delta.shape}")
print("Y_delta abs mean / max:", np.abs(Y_delta).mean(), np.abs(Y_delta).max())

# ── Normalisation pipeline (mirrors train_pinn.py) ──────────────
X_raw_tensor = torch.FloatTensor(X_combined.copy())
Y_raw_tensor = torch.FloatTensor(Y_delta)

perm_all = torch.randperm(len(X_raw_tensor))
X_all = X_raw_tensor[perm_all]
Y_all = Y_raw_tensor[perm_all]
n_train = int(TRAIN_SPLIT * len(X_all))
X_train_raw = X_all[:n_train]
Y_train_raw = Y_all[:n_train]

vel_indices = [3, 4, 5]
vel_q_low  = X_train_raw[:, vel_indices].quantile(0.01, dim=0)
vel_q_high = X_train_raw[:, vel_indices].quantile(0.99, dim=0)
X_train_raw[:, vel_indices] = torch.clamp(X_train_raw[:, vel_indices], vel_q_low, vel_q_high)

X_mean = X_train_raw.mean(dim=0)
X_std  = X_train_raw.std(dim=0) + 1e-8
X_train = (X_train_raw - X_mean) / X_std

print("\n--- Normalisation sanity ---")
zero_std_mask = (X_std < 1e-7)
print("Near-zero-std input columns:", zero_std_mask.sum().item(), "/", X_std.numel())
zero_std_idx = torch.nonzero(zero_std_mask).flatten().tolist()
col_names = base_cols + [f'nb_deform_{i}' for i in range(300)] + [f'nb_stress_{i}' for i in range(300)]
print("  -> indices/names:", [(i, col_names[i]) for i in zero_std_idx][:10])
print("X_train post-norm min/max/mean/std:", X_train.min().item(), X_train.max().item(), X_train.mean().item(), X_train.std().item())
nan_in_X = torch.isnan(X_train).sum().item()
inf_in_X = torch.isinf(X_train).sum().item()
print("NaN/Inf in X_train after norm:", nan_in_X, inf_in_X)

Y_mean = Y_train_raw.mean(dim=0)
Y_std  = Y_train_raw.std(dim=0) + 1e-8
Y_train = (Y_train_raw - Y_mean) / Y_std
print("Y_train post-norm min/max/mean/std:", Y_train.min().item(), Y_train.max().item(), Y_train.mean().item(), Y_train.std().item())

# ── Correlation: feature groups vs target (most active vertex) ──
print("\n--- Correlations with target delta (most-active vertex) ---")
dy_max = df[[c for c in df.columns if c.startswith('dy')]].abs().max()
best_vertex_id = None
for col in dy_max.sort_values(ascending=False).index:
    vid = int(col[2:])
    if vid not in FIXED_INDICES:
        best_vertex_id = vid
        break
print("Most active vertex:", best_vertex_id)
delta_best = dy_all[:, best_vertex_id] - pdy_all[:, best_vertex_id]

groups = {
    'tool_fx': df['tool_fx'].values,
    'tool_fy': df['tool_fy'].values,
    'tool_fz': df['tool_fz'].values,
    'tool_vx': df['tool_vx'].values,
    'tool_vy': df['tool_vy'].values,
    'sim_time': df['sim_time'].values,
    'dt_since_last': df['dt_since_last'].values,
    'fvy_at_best_vertex': df[f'fvy{best_vertex_id}'].values,
    'sax_at_best_vertex': df[f'sax{best_vertex_id}'].values,
    'say_at_best_vertex': df[f'say{best_vertex_id}'].values,
    'nb_deform_lag1[0]': nb_deform_features[:, 0],
    'nb_stress_lag1[0]': nb_stress_features[:, 0],
}
for name, vals in groups.items():
    if np.std(vals) < 1e-12:
        print(f"  {name:22s}: CONSTANT (std=0) -> corr undefined")
        continue
    r, _ = pearsonr(vals, delta_best)
    print(f"  {name:22s}: corr = {r:+.4f}")

print("\nDone.")
