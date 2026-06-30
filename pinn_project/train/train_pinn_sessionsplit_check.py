"""
PINN Training Script — Haptic Force Feedback Prediction (Final: history-only + real strain)
================================================================================
Predict, from PAST history only (lags 1..5, NO current-step info at all):
  - PRIMARY output (teaches physics): tissue deformation delta (178 active verts x 3) -> 534
  - GOAL output (reported L2):        haptic force felt by the device (tool_fx/fy/fz) -> 3
Total output: 537

Input (957):
  dt_cum_lag (cumulative time from base to this lag's row)      : 5
  dt_pred         (dt_since_last for the CURRENT step)           : 1
  current_pos_vel (tool_x,y,z, tool_vx,vy,vz AT current step)    : 6
  per lag in 1..5 (lag5 = oldest of the 5-row window, "base"):
    tool_x, tool_y, tool_z                                       : 3
    tool_vx, tool_vy, tool_vz                                    : 3
    tool_fx, tool_fy, tool_fz   (log1p compressed)               : 3
    nb_deform     (20 neighbours x dx,dy,dz delta)                : 60
    nb_stress     (20 neighbours x sax,say,saz - contact proxy)   : 60
    nb_realstrain (20 neighbours x rexx,reyy,rezz)                : 60
  -> 5 + 1 + 6 + (9+180)*5                                      = 957 total
  (current tool position/velocity + dt_pred re-added: instantaneous
   penetration depth/velocity strongly drives the haptic force output)

Loss: W_DEFORM * MSE(deform) + W_FORCE * MSE(force), separate per-column
z-score normalisation; deform teaches the bulk of the physics, force is the
reported headline metric.
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
from pinn_model import LagSequenceAttentionAccelVar

# ============================================================
# CONTACT-WEIGHT EXPERIMENT — same pipeline as the lag sweep, fixed at the
# already-established best n_lags=8, testing whether an extra loss weight for
# sustained-contact rows (separate from f_weight's force-magnitude weighting)
# improves sustained-hold accuracy. Appends one summary line per run to
# contactweight_results.txt for direct comparison against the n_lags=8 baseline.
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--n-lags', type=int, default=8)
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--sustain-beta', type=float, default=1.0)
args = parser.parse_args()

# ============================================================
# CONFIGURATION
# ============================================================
CSV_PATH      = f'{SOFA_ROOT}/pinn_project/data/training_data.csv'
N_VERTICES    = 181
FIXED_INDICES = [3, 39, 64]             # from FixedConstraint in scene
BATCH_SIZE    = 64
N_EPOCHS      = args.epochs
LR            = 3e-4
TRAIN_SPLIT   = 0.8
N_NEIGHBOURS  = 20
N_LAGS        = args.n_lags
W_FORCE       = 0.5
W_DEFORM      = 1.0
SEED          = 42

# Fix train/test split + weight init so runs are reproducible/comparable.
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# STEP 1: LOAD + FILTER DATA
# ============================================================
print("Loading data...")
df = pd.read_csv(CSV_PATH)

required_time_cols = {'sim_time', 'dt_since_last'}
missing_time_cols = required_time_cols - set(df.columns)
if missing_time_cols:
    raise ValueError(
        f"CSV is missing columns {missing_time_cols} — delete the old "
        f"{CSV_PATH} and recollect data with the updated DataCollector."
    )

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

force_mag_raw = np.linalg.norm(df[['tool_fx','tool_fy','tool_fz']].values, axis=1)
# keep ALL rows including non-contact — model must learn force=0 when not touching
# drop sessions shorter than N_LAGS*3 rows — too much zero-padding, pure noise
sess_counts = df.groupby('session_id')['session_id'].transform('count')
df = df[sess_counts >= N_LAGS * 3].reset_index(drop=True)
print(f"After short-session filter: {len(df)} rows")

max_disp = df[[c for c in df.columns if c.startswith('dx') or c.startswith('dy') or c.startswith('dz')]].values
pdmax    = df[[c for c in df.columns if c.startswith('pdx') or c.startswith('pdy') or c.startswith('pdz')]].values
delta_check = np.abs(max_disp - pdmax).max(axis=1)
df = df[delta_check < 2.0].reset_index(drop=True)

print(f"Rows after pre-filtering df: {len(df)}")

# CLIP FORCES SO DATA DISTRIBUTION MATCHES TRAINING (also clips the new force target)
df['tool_fx'] = df['tool_fx'].clip(-100.0, 100.0)
df['tool_fy'] = df['tool_fy'].clip(-100.0, 100.0)
df['tool_fz'] = df['tool_fz'].clip(-100.0, 100.0)

print(f"Rows after cleaning: {len(df)}")
N_rows = len(df)

# ============================================================
# Sustained-contact flag: rows that are part of a run of >=10 consecutive
# rows with |F|>3N, within the same session. f_weight (added later) already
# upweights by force MAGNITUDE, but a long, moderate-force hold doesn't get
# much boost from that — this is a separate axis (temporal under-representation
# of sustained contact, confirmed: ~72% of sessions have no hold >=5 rows).
# ============================================================
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

# Pick the most active vertex (skip fixed indices) — for correlation diagnostics
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

# ============================================================
# STEP 2: NEIGHBOURHOOD + LAG FEATURES (unchanged from existing pipeline)
# ============================================================
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
print(f"Neighbourhood computed. Sample neighbours row 0: {neighbour_idx[0]}")

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

# Real (Hooke's law) per-vertex strain from TetrahedronFEMForceField
rexx_all = df[[f'rexx{i}' for i in range(181)]].values
reyy_all = df[[f'reyy{i}' for i in range(181)]].values
rezz_all = df[[f'rezz{i}' for i in range(181)]].values

print("Building neighbourhood lag features...")
nb_deform_features    = np.zeros((N_rows, N_NEIGHBOURS * 3 * N_LAGS), dtype=np.float32)
nb_stress_features    = np.zeros((N_rows, N_NEIGHBOURS * 3 * N_LAGS), dtype=np.float32)
nb_realstrain_features = np.zeros((N_rows, N_NEIGHBOURS * 3 * N_LAGS), dtype=np.float32)
session_ids = df['session_id'].values

for row in range(N_rows):
    nb = neighbour_idx[row]
    deform_row, stress_row, realstrain_row = [], [], []
    for lag in range(1, N_LAGS + 1):
        src_row = max(0, row - lag)
        if session_ids[src_row] != session_ids[row]:
            # pad with zeros — crossed session boundary, no valid history
            deform_row.append(np.zeros(N_NEIGHBOURS))
            deform_row.append(np.zeros(N_NEIGHBOURS))
            deform_row.append(np.zeros(N_NEIGHBOURS))
            stress_row.append(np.zeros(N_NEIGHBOURS))
            stress_row.append(np.zeros(N_NEIGHBOURS))
            stress_row.append(np.zeros(N_NEIGHBOURS))
            realstrain_row.append(np.zeros(N_NEIGHBOURS))
            realstrain_row.append(np.zeros(N_NEIGHBOURS))
            realstrain_row.append(np.zeros(N_NEIGHBOURS))
        else:
            deform_row.append(ddx_all[src_row][nb])
            deform_row.append(ddy_all[src_row][nb])
            deform_row.append(ddz_all[src_row][nb])
            stress_row.append(sax_all[src_row][nb])
            stress_row.append(say_all[src_row][nb])
            stress_row.append(saz_all[src_row][nb])
            realstrain_row.append(rexx_all[src_row][nb])
            realstrain_row.append(reyy_all[src_row][nb])
            realstrain_row.append(rezz_all[src_row][nb])
    nb_deform_features[row]     = np.concatenate(deform_row)
    nb_stress_features[row]     = np.concatenate(stress_row)
    nb_realstrain_features[row] = np.concatenate(realstrain_row)

print(f"Neighbourhood deform features shape:      {nb_deform_features.shape}")
print(f"Neighbourhood stress features shape:      {nb_stress_features.shape}")
print(f"Neighbourhood real-strain features shape: {nb_realstrain_features.shape}")

from scipy.stats import pearsonr
delta_best = dy_all[:, best_vertex_id] - pdy_all[:, best_vertex_id]
corr_deform, _ = pearsonr(nb_deform_features[:, 0], delta_best)
corr_stress, _ = pearsonr(nb_stress_features[:, 0], delta_best)
corr_realstrain, _ = pearsonr(nb_realstrain_features[:, 0], delta_best)
print(f"Neighbour delta deform lag1 corr with target delta:    {corr_deform:.3f}")
print(f"Neighbour stress lag1 corr with target delta:          {corr_stress:.3f}")
print(f"Neighbour real-strain lag1 corr with target delta:     {corr_realstrain:.3f}")

# ============================================================
# STEP 3: TOOL KINEMATIC HISTORY (lags 1..5, 9 values each) — no current row
# ============================================================
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
# Start index of the current contiguous run of this session_id (NOT a global
# searchsorted — session_id=0 (no-contact) recurs between numbered episodes,
# so the column isn't sorted ascending and searchsorted can return bogus/OOB
# indices once there are enough interleaved runs).
run_start_idx = np.zeros(N_rows, dtype=np.int64)
for row in range(1, N_rows):
    run_start_idx[row] = row if session_ids[row] != session_ids[row - 1] else run_start_idx[row - 1]

for row in range(N_rows):
    lag_vals = []
    sess_start = run_start_idx[row]
    for lag in range(1, N_LAGS + 1):
        src_row = max(0, row - lag)
        if session_ids[src_row] != session_ids[row]:
            # pad: repeat session-start position, zero velocity and force
            lag_vals.append([
                tool_x_all[sess_start], tool_y_all[sess_start], tool_z_all[sess_start],
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
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

# ============================================================
# STEP 3b: CURRENT-STEP FEATURES — dt_pred (1) + current_pos_vel (6)
# ============================================================
dt_pred_vals = np.zeros((N_rows, 1), dtype=np.float32)
for row in range(N_rows):
    base_row = max(0, row - N_LAGS)
    dt_pred_vals[row, 0] = real_time_vals[row] - real_time_vals[base_row]
dt_pred_vals = np.clip(dt_pred_vals, 0.0, 0.15)
current_pos_vel = np.stack([
    tool_x_all, tool_y_all, tool_z_all,
    tool_vx_all, tool_vy_all, tool_vz_all,
], axis=1).astype(np.float32)  # (N, 6)

# Contact detection features (3):
#   min_dist_to_mesh  — how close is tool tip to nearest liver vertex RIGHT NOW
#   delta_min_dist    — is tool approaching (neg) or pulling away (pos)?
#   lag1_contact_flag — was force>0.01 last step? (binary onset signal, no leakage)
delta_min_dist = np.zeros(N_rows, dtype=np.float32)
delta_min_dist[1:] = min_dist_to_mesh[1:] - min_dist_to_mesh[:-1]

force_mag_all = np.linalg.norm(
    df[['tool_fx', 'tool_fy', 'tool_fz']].values, axis=1
).astype(np.float32)
lag1_contact = np.zeros(N_rows, dtype=np.float32)
lag1_contact[1:] = (force_mag_all[:-1] > 0.01).astype(np.float32)

contact_features = np.stack(
    [min_dist_to_mesh, delta_min_dist, lag1_contact], axis=1
).astype(np.float32)  # (N, 3)

# Acceleration (wider lag1-vs-lag3 baseline, NOT adjacent-step — validated via
# correlation test: leading corr(accel_t, |dF| over next 3 steps) = 0.66, the
# strongest predictive feature found so far. Adjacent-step differencing would
# amplify position-sensor noise; this wider baseline averages it out while still
# catching genuine sustained acceleration from fast flicks.
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

corr_dt_pred,    _ = pearsonr(dt_pred_vals[:, 0], delta_best)
corr_min_dist,   _ = pearsonr(min_dist_to_mesh,   force_mag_all)
corr_delta_dist, _ = pearsonr(delta_min_dist,      force_mag_all)
corr_lag1_cont,  _ = pearsonr(lag1_contact,        force_mag_all)
accel_mag = np.linalg.norm(accel, axis=1)
corr_accel, _ = pearsonr(accel_mag, force_mag_all)
print(f"accel_mag corr with force_mag (simultaneous):       {corr_accel:.3f}")
print(f"dt_pred corr with target delta:                     {corr_dt_pred:.3f}")
print(f"min_dist_to_mesh corr with force_mag:               {corr_min_dist:.3f}")
print(f"delta_min_dist corr with force_mag:                 {corr_delta_dist:.3f}")
print(f"lag1_contact_flag corr with force_mag:              {corr_lag1_cont:.3f}")

# ============================================================
# STEP 4: CUMULATIVE dt FEATURES — dt_cum (5)
# ============================================================
# base_row = row - N_LAGS (clamped to 0): the oldest row in the 5-row window.
# dt_cum[lag] = real_time[src_row(lag)] - real_time[base_row]   (lag5 == 0 by def.)
# Clipped to [0, 0.15s] to absorb session-boundary artifacts (same spirit as dt_window).
dt_cum  = np.zeros((N_rows, N_LAGS), dtype=np.float32)
for row in range(N_rows):
    base_row = max(0, row - N_LAGS)
    rt_base = real_time_vals[base_row]
    for lag in range(1, N_LAGS + 1):
        src_row = max(0, row - lag)
        if session_ids[src_row] != session_ids[row]:
            dt_cum[row, lag - 1] = 0.0  # no time history before session start
        else:
            dt_cum[row, lag - 1] = real_time_vals[src_row] - rt_base

dt_cum  = np.clip(dt_cum,  0.0, 0.15)

# ============================================================
# STEP 5: ASSEMBLE X (957) — current-step features + history lags 1..5
# ============================================================
X_combined = np.concatenate([
    dt_cum,                       # (N, 5)   idx   0-4
    dt_pred_vals,                 # (N, 1)   idx   5
    current_pos_vel,              # (N, 6)   idx   6-11
    contact_features,             # (N, 3)   idx  12-14  [min_dist, delta_dist, lag1_contact]
    tool_hist_features,           # (N, 45)  idx  15-59
    nb_deform_features,           # (N, 300) idx  60-359
    nb_stress_features,           # (N, 300) idx 360-659
    nb_realstrain_features,       # (N, 300) idx 660-959
    accel,                        # (N, 3)   idx  960-962  NEW — lag1-vs-lag3 acceleration
], axis=1).astype(np.float32)      # (N, 963)

N_INPUTS = X_combined.shape[1]
X_raw_tensor = torch.FloatTensor(X_combined.copy())
print(f"Total inputs: {N_INPUTS}")  # should print 963

# Velocity column indices (for outlier clamping):
# current tool_vx/vy/vz (idx 9-11) + tool_vx/vy/vz at each of the 5 history lags
# contact_features occupy idx 12-14; tool_hist starts at 15; accel at the new tail 960-962
vel_indices = [9, 10, 11]
for k in range(N_LAGS):
    base = 15 + 9 * k          # start of lag-k's 9-value block within tool_hist_features
    vel_indices += [base + 3, base + 4, base + 5]
vel_indices += [960, 961, 962]

# ============================================================
# STEP 7: ASSEMBLE Y (537) — force (3, primary) + deformation delta (534)
# ============================================================
active_verts = [i for i in range(N_VERTICES) if i not in FIXED_INDICES]
N_OUT_DEFORM = len(active_verts) * 3  # 534

dy_cols, pdy_cols = [], []
for i in active_verts:
    dy_cols  += [f'dx{i}',  f'dy{i}',  f'dz{i}']
    pdy_cols += [f'pdx{i}', f'pdy{i}', f'pdz{i}']

Y_deform = (df[dy_cols].values - df[pdy_cols].values).astype(np.float32)  # (N, 534)

# Force target — log1p-compressed (same transform as the force INPUT features);
# inverted at eval time to report errors in real Newtons.
force_raw = df[['tool_fx','tool_fy','tool_fz']].values.astype(np.float32)  # (N, 3)
Y_force = np.sign(force_raw) * np.log1p(np.abs(force_raw))                 # (N, 3)

N_FORCE = 3
Y_combined = np.concatenate([Y_force, Y_deform], axis=1).astype(np.float32)  # (N, 537)
N_OUT = Y_combined.shape[1]
Y_raw_tensor = torch.FloatTensor(Y_combined.copy())
print(f"Total outputs: {N_OUT} (force={N_FORCE}, deform={N_OUT_DEFORM})")

# ============================================================
# STEP 8: TRAIN/TEST SPLIT — SESSION-LEVEL HOLDOUT (sanity check variant)
# ============================================================
# Original script split ROWS randomly (see git history), which lets rows from
# the same touch sit in both train and test — neighbouring rows in a session
# are near-duplicates, so that "test" error mostly measures interpolation
# between seen rows, not generalisation to a genuinely new touch. Here we
# instead hold out entire sessions (session_id != 0) so test rows come ONLY
# from touches the model never saw any part of — a fair generalisation check,
# directly comparable against the original row-random number.
rng = np.random.RandomState(SEED)
all_sids = np.unique(session_ids)
contact_sids = all_sids[all_sids != 0]
rng.shuffle(contact_sids)
n_test_sids = max(1, int(round((1 - TRAIN_SPLIT) * len(contact_sids))))
test_sids = set(contact_sids[:n_test_sids].tolist())
print(f"  Held-out sessions ({len(test_sids)}/{len(contact_sids)}): {sorted(test_sids)}")

is_test_row = np.array([sid in test_sids for sid in session_ids])
# session_id==0 (no-contact filler) rows aren't a coherent "touch" — split those
# row-randomly as before so they don't all land on one side.
zero_idx = np.where(session_ids == 0)[0]
rng.shuffle(zero_idx)
n_zero_test = int(round((1 - TRAIN_SPLIT) * len(zero_idx)))
zero_test_set = set(zero_idx[:n_zero_test].tolist())
for i in zero_idx:
    is_test_row[i] = i in zero_test_set

test_idx_np  = np.where(is_test_row)[0]
train_idx_np = np.where(~is_test_row)[0]
test_idx  = torch.LongTensor(test_idx_np)
train_idx = torch.LongTensor(train_idx_np)

X_train_raw = X_raw_tensor[train_idx]
Y_train_raw = Y_raw_tensor[train_idx]
X_test_raw  = X_raw_tensor[test_idx]
Y_test_raw  = Y_raw_tensor[test_idx]
sustained_train_raw = sustained_tensor_all[train_idx]
sustained_test_raw  = sustained_tensor_all[test_idx]

perm = torch.randperm(len(X_train_raw))
X_train_raw = X_train_raw[perm]
Y_train_raw = Y_train_raw[perm]
sustained_train_raw = sustained_train_raw[perm]

print(f"  Input  shape: {X_train_raw.shape}")
print(f"  Output shape: {Y_train_raw.shape}")
print(f"  Train: {len(X_train_raw)}  |  Test: {len(X_test_raw)}")

# Z-SCORE TARGET NORMALISATION (per-column — force and deform get separate stats)
Y_mean = Y_train_raw.mean(dim=0)
Y_std  = Y_train_raw.std(dim=0) + 1e-8
Y_train = (Y_train_raw - Y_mean) / Y_std
Y_test  = (Y_test_raw  - Y_mean) / Y_std
print(f"  Target Z-Score Stats -> Mean: {Y_train.mean().item():.6f} | Std: {Y_train.std().item():.6f}")

# ── Normalize inputs ──────────────────────────────────────────
vel_idx_t = torch.tensor(vel_indices)
vel_q_low  = X_train_raw[:, vel_idx_t].quantile(0.01, dim=0)
vel_q_high = X_train_raw[:, vel_idx_t].quantile(0.99, dim=0)
X_train_raw[:, vel_idx_t] = torch.clamp(X_train_raw[:, vel_idx_t], vel_q_low, vel_q_high)
X_test_raw[:,  vel_idx_t] = torch.clamp(X_test_raw[:,  vel_idx_t], vel_q_low, vel_q_high)

X_mean = X_train_raw.mean(dim=0)
X_std  = X_train_raw.std(dim=0) + 1e-8
X_train = (X_train_raw - X_mean) / X_std
X_test  = (X_test_raw  - X_mean) / X_std

print(f"  X std min/max: {X_std.min().item():.6f} / {X_std.max().item():.6f}")

# ============================================================
# STEP 9: TRAINING
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

model = LagSequenceAttentionAccelVar(n_output=N_OUT, n_inputs=N_INPUTS, n_lags=N_LAGS).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=1000, T_mult=2, eta_min=1e-6
)

SUSTAIN_BETA = args.sustain_beta  # extra weight multiplier for sustained-contact rows

dataset  = TensorDataset(X_train.to(device), Y_train.to(device), sustained_train_raw.to(device))
loader   = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
y_std_f  = Y_std[:N_FORCE].detach().clone().to(device)
y_mean_f = Y_mean[:N_FORCE].detach().clone().to(device)

history = {'epoch': [], 'total': [], 'force': [], 'deform': []}

best_val_loss = float('inf')
best_epoch    = 0
MODEL_BEST    = f'tissue_pinn_sessionsplit_n{N_LAGS}_best.pth'
EARLY_STOP_PATIENCE = 800  # epochs without val improvement before stopping

print("\nTraining...\n")
for epoch in range(N_EPOCHS):
    model.train()
    epoch_total = epoch_force = epoch_deform = 0.0

    for X_batch, Y_batch, sustain_batch in loader:
        u_pred = model(X_batch)

        # weighted force loss — upweight high-force samples (rare but critical)
        # AND sustained-contact samples (rare in a different way: not magnitude,
        # but temporal under-representation — see is_sustained computation above).
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

    # checkpoint best model using val set
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
            print(f"\nEarly stopping at epoch {epoch} — no val improvement in {EARLY_STOP_PATIENCE} epochs "
                  f"(best was epoch {best_epoch}, val loss {best_val_loss:.6f})")
            break

    if epoch % 200 == 0:
        history['epoch'].append(epoch)
        history['total'].append(epoch_total / n_batches)
        history['force'].append(epoch_force / n_batches)
        history['deform'].append(epoch_deform / n_batches)
        print(f"Epoch {epoch:5d} | "
              f"Total: {epoch_total/n_batches:.6f} | "
              f"Force: {epoch_force/n_batches:.6f} | "
              f"Deform: {epoch_deform/n_batches:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

print(f"\nBest checkpoint: epoch {best_epoch} (val loss {best_val_loss:.6f}) → {MODEL_BEST}")
model.load_state_dict(torch.load(MODEL_BEST, map_location=device))

# ============================================================
# STEP 10: VALIDATE
# ============================================================
def inv_log1p_signed(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

print("\n=== Validation vs FEM ===")
model.eval()
with torch.no_grad():
    u_pred_norm = model(X_test.to(device)).cpu()
    u_pred = (u_pred_norm * Y_std) + Y_mean
    u_true = (Y_test * Y_std) + Y_mean

    # ---- FORCE metrics (primary) — invert log1p back to Newtons ----
    force_pred_log = u_pred[:, :N_FORCE]
    force_true_log = u_true[:, :N_FORCE]
    force_pred_N = inv_log1p_signed(force_pred_log)
    force_true_N = inv_log1p_signed(force_true_log)

    force_rel_err = torch.norm(force_pred_N - force_true_N) / (torch.norm(force_true_N) + 1e-8)
    force_mae  = (force_pred_N - force_true_N).abs().mean()
    force_rmse = torch.sqrt(((force_pred_N - force_true_N) ** 2).mean())
    per_sample_force_err = torch.norm(force_pred_N - force_true_N, dim=1)

    # Robust variant: relative L2 excluding the worst ~1% of samples. The log1p/exp
    # inverse transform exponentially amplifies single-sample z-score misses, so one
    # hard test sample can dominate the plain relative-L2 and make a good model look broken.
    n_test = len(per_sample_force_err)
    k_outliers = max(1, n_test // 100)
    err_thresh = torch.topk(per_sample_force_err, k_outliers).values.min()
    robust_mask = per_sample_force_err < err_thresh
    force_rel_err_robust = torch.norm(force_pred_N[robust_mask] - force_true_N[robust_mask]) / \
                           (torch.norm(force_true_N[robust_mask]) + 1e-8)

    # Sustained-contact-specific error — the actual thing this experiment targets,
    # not just overall force error (which sustained rows are a minority of).
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
    print(f"MAE (N):                    {force_mae.item():.6f}")
    print(f"RMSE (N):                   {force_rmse.item():.6f}")
    print(f"Median |force| error (N):   {per_sample_force_err.median().item():.6f}")
    print(f"Mean |force| error (N):     {per_sample_force_err.mean().item():.6f}")
    print(f"Max  |force| error (N):     {per_sample_force_err.max().item():.6f}")
    print(f"Test true |force| mean (N): {torch.norm(force_true_N, dim=1).mean().item():.6f}")

    # ---- DEFORMATION metrics (secondary) ----
    deform_pred = u_pred[:, N_FORCE:]
    deform_true = u_true[:, N_FORCE:]
    deform_rel_err = torch.norm(deform_pred - deform_true) / (torch.norm(deform_true) + 1e-8)

    N_ACTIVE = N_VERTICES - len(FIXED_INDICES)  # 178
    diff = (deform_pred - deform_true).reshape(-1, N_ACTIVE, 3)
    per_vertex_err = torch.norm(diff, dim=2)
    max_err_mm  = per_vertex_err.max().item() * 1000
    mean_err_mm = per_vertex_err.mean().item() * 1000
    within_1mm = (per_vertex_err.max(dim=1).values < 0.001).float().mean().item()

    print(f"\n--- DEFORMATION (secondary output) ---")
    print(f"Relative L2 error (deform): {deform_rel_err.item()*100:.2f}%")
    print(f"Max vertex error:           {max_err_mm:.3f} mm")
    print(f"Mean vertex error:          {mean_err_mm:.4f} mm")
    print(f"Samples within 1mm:         {within_1mm*100:.1f}%")

    # ---- Sample comparison plot (force, since that's the primary output) ----
    import matplotlib
    matplotlib.use('Agg')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(force_true_N[:200, 0].numpy(), label='FEM true fx', alpha=0.7)
    axes[0].plot(force_pred_N[:200, 0].numpy(), label='PINN pred fx', alpha=0.7)
    axes[0].set_title('Force fx — first 200 test samples')
    axes[0].legend()

    worst_idx = per_sample_force_err.argmax().item()
    best_idx  = per_sample_force_err.argmin().item()
    axes[1].bar(['fx','fy','fz'], force_true_N[worst_idx].numpy(), alpha=0.5, label='true (worst sample)')
    axes[1].bar(['fx','fy','fz'], force_pred_N[worst_idx].numpy(), alpha=0.5, label='pred (worst sample)')
    axes[1].set_title(f'Worst sample ({worst_idx}) force comparison')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('sample_comparison_sessionsplit.png', dpi=150)
    print("\nSample comparison saved to sample_comparison_sessionsplit.png")

# ============================================================
# STEP 11: SAVE
# ============================================================
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
}, f'tissue_pinn_sessionsplit_n{N_LAGS}_beta{SUSTAIN_BETA}.pth')
print(f"Model saved to tissue_pinn_sessionsplit_n{N_LAGS}_beta{SUSTAIN_BETA}.pth")

with open('sessionsplit_results.txt', 'a') as f:
    f.write(f"n_lags={N_LAGS:3d}  sustain_beta={SUSTAIN_BETA:.2f}  epochs_run={best_epoch:5d}  "
            f"best_val_loss={best_val_loss:.6f}  "
            f"force_rel_l2={force_rel_err.item()*100:6.2f}%  "
            f"force_rel_l2_robust={force_rel_err_robust.item()*100:6.2f}%  "
            f"force_rel_l2_sustained={force_rel_err_sustained.item()*100:6.2f}%  "
            f"deform_rel_l2={deform_rel_err.item()*100:6.2f}%  "
            f"held_out_sessions={sorted(test_sids)}  "
            f"params={n_params:,}\n")
print("Summary appended to sessionsplit_results.txt")

# ============================================================
# STEP 12: PLOT LOSS CURVES
# ============================================================
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history['epoch'], history['total'],  label='Total (weighted)')
plt.plot(history['epoch'], history['force'],  label='Force MSE')
plt.plot(history['epoch'], history['deform'], label='Deform MSE')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.title('Training Loss')
plt.yscale('log')

plt.subplot(1, 2, 2)
plt.bar(['Force Rel L2 %', 'Deform Rel L2 %'],
        [force_rel_err.item()*100, deform_rel_err.item()*100])
plt.title('Validation vs FEM')

plt.tight_layout()
plt.savefig('training_results_sessionsplit.png', dpi=150)
print("Plot saved to training_results_sessionsplit.png")
