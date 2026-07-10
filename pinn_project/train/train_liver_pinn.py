"""
Liver PINN — unified overnight architecture search script.

Usage examples:
  python train_liver_pinn.py --run-id R1_baseline
  python train_liver_pinn.py --run-id R2_no_accstress --no-accstress
  python train_liver_pinn.py --run-id R3_add_vel --add-vel
  python train_liver_pinn.py --run-id R4_n4 --n-lags 4
  python train_liver_pinn.py --run-id R5_n16 --n-lags 16
  python train_liver_pinn.py --run-id R6_time1s --time-window 1.0
  python train_liver_pinn.py --run-id R7_time2s --time-window 2.0
  python train_liver_pinn.py --run-id R8_k10 --n-neighbours 10
  python train_liver_pinn.py --run-id R9_stress --no-strain --use-stress
  python train_liver_pinn.py --run-id R10_dual --arch dual_attn

Feature layout (all n_nb_feat options enabled = 15 per neighbour):
  Global (n_lags+13): dt_cum(n_lags), dt_pred(1), pos_vel(6), contact(3), accel(3)
  Per-lag × n_lags:   tool(9) + nb_concat(K × n_nb_feat)
    n_nb_feat = 3(deform) + 3?(accstress) + 3?(strain) + 3?(vel) + 3?(stress)
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
from pinn_model import LiverSeqAttnFlex, LiverDualAttnFlex, LiverDualAttnVarLen

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--run-id', type=str, default=None)
parser.add_argument('--n-lags', type=int, default=8)
parser.add_argument('--n-neighbours', type=int, default=20)
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--sustain-beta', type=float, default=5.0)
parser.add_argument('--arch', type=str, default='dual_attn',
                    choices=['seq_attn', 'dual_attn'])
# feature toggles (accstress and strain ON by default, matching 200k best)
parser.add_argument('--no-accstress', action='store_true', help='Exclude accStress from nb features')
parser.add_argument('--no-strain',    action='store_true', help='Exclude real strain from nb features')
parser.add_argument('--add-vel',      action='store_true', help='Include vertex velocities in nb features')
parser.add_argument('--use-stress',   action='store_true', help='Include diagonal FEM stress in nb features')
# time-based window (0 = row-based lags)
parser.add_argument('--time-window', type=float, default=0.0,
                    help='If >0, use time-based fixed window of this many seconds')
parser.add_argument('--n-time-points', type=int, default=8,
                    help='Number of lookup points in time window')
parser.add_argument('--var-window', type=float, default=0.0,
                    help='If >0, use ALL rows within this many seconds as variable-length history')
parser.add_argument('--max-rows', type=int, default=0,
                    help='If >0, load only the first N rows of the CSV (for quick tests)')
# output mode — force-only and filter-short are on by default (best practice)
parser.add_argument('--force-only', action='store_true', default=True,
                    help='Predict only force (3 outputs), skip deformation head')
parser.add_argument('--no-force-only', dest='force_only', action='store_false')
parser.add_argument('--filter-short', action='store_true', default=True,
                    help='Drop sessions with fewer than 30 rows (noisy taps)')
parser.add_argument('--no-filter-short', dest='filter_short', action='store_false')
# loss improvements
parser.add_argument('--high-force-gamma', type=float, default=1.0,
                    help='Power for force-magnitude loss weighting (1=linear, 2=quadratic). '
                         'Higher values upweight high-force errors to fix under-prediction.')
# global feature additions
parser.add_argument('--session-progress', action='store_true',
                    help='Add session progress (0=onset, 1=end) to global features')
# neighbour feature additions
parser.add_argument('--add-normals', action='store_true',
                    help='Add precomputed vertex normals (nx,ny,nz) to neighbour features')
# E generalisation
parser.add_argument('--use-youngs', action='store_true',
                    help='Add youngs_modulus as a global input feature (requires E-sweep CSV)')
parser.add_argument('--csv-path', type=str, default=None,
                    help='Override default training CSV path (default: data/training_data.csv)')
args = parser.parse_args()

import glob as _glob
def _auto_run_id(use_youngs, time_window, var_window):
    if var_window > 0:
        vw = f'vw{var_window:.1f}s'.replace('.','p')
        prefix = f'Egen_{vw}' if use_youngs else f'E1500_{vw}'
    elif time_window > 0:
        tw = f'tw{time_window:.1f}s'.replace('.','p')
        prefix = f'Egen_{tw}' if use_youngs else f'E1500_{tw}'
    else:
        prefix = 'Egen' if use_youngs else 'E1500'
    existing = _glob.glob(f'{SOFA_ROOT}/pinn_project/train/liver_{prefix}_v*.pth')
    nums = []
    for p in existing:
        base = p.split(f'_{prefix}_v')[-1].replace('.pth','')
        if base.isdigit(): nums.append(int(base))
    return f'{prefix}_v{max(nums)+1 if nums else 1}'
RUN_ID = args.run_id if args.run_id else _auto_run_id(args.use_youngs, args.time_window, args.var_window)
N_LAGS        = args.n_lags
N_NEIGHBOURS  = args.n_neighbours
N_EPOCHS      = args.epochs
SUSTAIN_BETA  = args.sustain_beta
ARCH          = args.arch
USE_ACCSTRESS = not args.no_accstress
USE_STRAIN    = not args.no_strain
USE_VEL       = args.add_vel
USE_STRESS    = args.use_stress
TIME_WINDOW   = args.time_window
N_TIME_POINTS = args.n_time_points
VAR_WINDOW    = args.var_window
MAX_STEPS_VAR = int(VAR_WINDOW / 0.005) if VAR_WINDOW > 0 else 0  # e.g. 200 for 1.0s at 5ms/step
FORCE_ONLY        = args.force_only
FILTER_SHORT      = args.filter_short
USE_YOUNGS        = args.use_youngs
CSV_PATH_OVERRIDE = args.csv_path
HIGH_FORCE_GAMMA  = args.high_force_gamma
SESSION_PROGRESS  = args.session_progress
ADD_NORMALS       = args.add_normals

# For time-based mode, N_LAGS is re-used as the number of lookup points
N_STEPS       = N_TIME_POINTS if TIME_WINDOW > 0 else N_LAGS

n_nb_feat = 3 + 3*USE_ACCSTRESS + 3*USE_STRAIN + 3*USE_VEL + 3*USE_STRESS + 3*ADD_NORMALS
print(f"\nRun: {RUN_ID}")
print(f"arch={ARCH}  n_steps={N_STEPS}  K={N_NEIGHBOURS}  n_nb_feat={n_nb_feat}")
print(f"features: deform(always) accstress={USE_ACCSTRESS} strain={USE_STRAIN} vel={USE_VEL} stress={USE_STRESS} normals={ADD_NORMALS}")
print(f"time_window={'%.1fs'%TIME_WINDOW if TIME_WINDOW>0 else 'row-based'}  "
      f"high_force_gamma={HIGH_FORCE_GAMMA}  session_progress={SESSION_PROGRESS}\n")

if CSV_PATH_OVERRIDE:
    CSV_PATH = CSV_PATH_OVERRIDE
elif USE_YOUNGS:
    CSV_PATH = f'{SOFA_ROOT}/pinn_project/data/training_data_E_gen_30k_per_E_v2.csv'
else:
    CSV_PATH = f'{SOFA_ROOT}/pinn_project/data/training_data.csv'
RESULTS_FILE = f'{SOFA_ROOT}/pinn_project/train/liver_results.txt'
N_VERTICES   = 181
FIXED_INDICES= [3, 39, 64]
BATCH_SIZE   = 64
LR           = 3e-4
TRAIN_SPLIT  = 0.8
W_FORCE      = 0.5
W_DEFORM     = 1.0
SEED         = 42
EARLY_STOP   = 800

torch.manual_seed(SEED); np.random.seed(SEED)

# ── Load + clean ──────────────────────────────────────────────────────────────
print("Loading data...")
_nrows = args.max_rows if args.max_rows > 0 else None
df = pd.read_csv(CSV_PATH, nrows=_nrows, dtype={c: 'float32' for c in pd.read_csv(CSV_PATH, nrows=0).columns
                                  if c not in ('step','session_id')})
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

# ── Sustained-contact flag ────────────────────────────────────────────────────
session_ids = df['session_id'].values
df_fmag = np.linalg.norm(df[['tool_fx','tool_fy','tool_fz']].values, axis=1)
is_sustained = np.zeros(N_rows, dtype=np.float32)
for sid in np.unique(session_ids):
    if sid == 0: continue
    idx = df.index[df['session_id'] == sid].values
    high = df_fmag[idx] > 3.0
    run_id = np.zeros(len(high), dtype=int)
    cur = 0
    for i in range(len(high)):
        cur = cur + 1 if high[i] else 0
        run_id[i] = cur
    if run_id.max() >= 10:
        is_sustained[idx[high]] = 1.0
print(f"Sustained-contact rows: {int(is_sustained.sum())} / {N_rows} "
      f"({100*is_sustained.sum()/N_rows:.1f}%)")

# ── Load raw arrays ───────────────────────────────────────────────────────────
vertex_pos = np.load(f'{SOFA_ROOT}/pinn_project/data/liver_vertices.npy')
tool_pos   = df[['tool_x','tool_y','tool_z']].values
real_time  = df['real_time'].values
sim_time   = df['sim_time'].values

# Precompute vertex normals via PCA on local mesh neighbourhood (no faces needed)
if ADD_NORMALS:
    print("Precomputing vertex normals from mesh point cloud...")
    vertex_normals = np.zeros((len(vertex_pos), 3), dtype=np.float32)
    for vi in range(len(vertex_pos)):
        dists = np.linalg.norm(vertex_pos - vertex_pos[vi], axis=1)
        nn_idx = np.argsort(dists)[1:13]          # 12 nearest mesh neighbours
        pts = vertex_pos[nn_idx] - vertex_pos[vi]
        _, _, Vt = np.linalg.svd(pts, full_matrices=False)
        n = Vt[-1]                                 # eigenvector with smallest singular value
        vertex_normals[vi] = n / (np.linalg.norm(n) + 1e-8)
    print(f"Vertex normals computed, shape: {vertex_normals.shape}")

dx_all  = df[[f'dx{i}'  for i in range(181)]].values
dy_all  = df[[f'dy{i}'  for i in range(181)]].values
dz_all  = df[[f'dz{i}'  for i in range(181)]].values
pdx_all = df[[f'pdx{i}' for i in range(181)]].values
pdy_all = df[[f'pdy{i}' for i in range(181)]].values
pdz_all = df[[f'pdz{i}' for i in range(181)]].values

ddx_all = dx_all - pdx_all
ddy_all = dy_all - pdy_all
ddz_all = dz_all - pdz_all

if USE_ACCSTRESS:
    sax_all = df[[f'sax{i}' for i in range(181)]].values
    say_all = df[[f'say{i}' for i in range(181)]].values
    saz_all = df[[f'saz{i}' for i in range(181)]].values

if USE_STRAIN:
    rexx_all = df[[f'rexx{i}' for i in range(181)]].values
    reyy_all = df[[f'reyy{i}' for i in range(181)]].values
    rezz_all = df[[f'rezz{i}' for i in range(181)]].values

if USE_VEL:
    vvx_all = df[[f'vvx{i}' for i in range(181)]].values
    vvy_all = df[[f'vvy{i}' for i in range(181)]].values
    vvz_all = df[[f'vvz{i}' for i in range(181)]].values

if USE_STRESS:
    rsxx_all = df[[f'rsxx{i}' for i in range(181)]].values
    rsyy_all = df[[f'rsyy{i}' for i in range(181)]].values
    rszz_all = df[[f'rszz{i}' for i in range(181)]].values

tool_x_all  = df['tool_x'].values;  tool_y_all = df['tool_y'].values
tool_z_all  = df['tool_z'].values
tool_vx_all = df['tool_vx'].values; tool_vy_all = df['tool_vy'].values
tool_vz_all = df['tool_vz'].values
tool_fx_all = df['tool_fx'].values; tool_fy_all = df['tool_fy'].values
tool_fz_all = df['tool_fz'].values

def log1p_signed(v):
    return np.sign(v) * np.log1p(np.abs(v))

# ── Neighbourhood indices ─────────────────────────────────────────────────────
print("Computing neighbourhood indices...")
neighbour_idx    = np.zeros((N_rows, N_NEIGHBOURS), dtype=int)
min_dist_to_mesh = np.zeros(N_rows, dtype=np.float32)
for row in range(N_rows):
    dists = np.linalg.norm(vertex_pos - tool_pos[row], axis=1)
    idx_s = np.argsort(dists)
    neighbour_idx[row]    = idx_s[:N_NEIGHBOURS]
    min_dist_to_mesh[row] = dists[idx_s[0]]

# ── Time-based window: pre-build per-session row lookup ───────────────────────
if TIME_WINDOW > 0:
    time_offsets = np.linspace(0.005, TIME_WINDOW, N_TIME_POINTS)  # 5ms (1 step) → TIME_WINDOW
    print(f"Time offsets: {[f'{t*1000:.0f}ms' for t in time_offsets]}")
    # FIX: use sim_time (constant 5ms/step) not real_time (variable wall-clock FPS)
    session_row_map = {}
    for sid in np.unique(session_ids):
        rows_s = np.where(session_ids == sid)[0]
        session_row_map[sid] = (rows_s, sim_time[rows_s])

    def find_src_row(row, offset):
        sid = session_ids[row]
        rows_s, sts_s = session_row_map[sid]
        target = sim_time[row] - offset           # sim_time: constant 5ms steps
        if target <= sts_s[0]:
            return rows_s[0]
        pos = np.searchsorted(sts_s, target, side='right') - 1
        return rows_s[max(0, min(pos, len(rows_s)-1))]

    def get_src_rows_for_row(row):
        return [find_src_row(row, t) for t in time_offsets]
else:
    def get_src_rows_for_row(row):
        return [max(0, row - (k+1)) for k in range(N_LAGS)]

# ── Build per-lag neighbourhood features ──────────────────────────────────────
# Layout per lag k: [tool(9), nb_0_feat, nb_1_feat, ..., nb_{K-1}_feat]
# n_nb_feat values per neighbour
lag_w = 9 + N_NEIGHBOURS * n_nb_feat
n_global = N_STEPS + 13 + int(SESSION_PROGRESS) + int(USE_YOUNGS)  # dt_cum + dt_pred + pos_vel + contact + accel [+ sess_progress] [+ E]

print(f"Building lag features ({N_STEPS} steps, {lag_w} per step, {n_global} global)...")
X_lags = np.zeros((N_rows, N_STEPS * lag_w), dtype=np.float32)

# Track run-start to correctly zero-pad at session boundaries
run_start_idx = np.zeros(N_rows, dtype=np.int64)
for row in range(1, N_rows):
    run_start_idx[row] = row if session_ids[row] != session_ids[row-1] else run_start_idx[row-1]

for row in range(N_rows):
    nb = neighbour_idx[row]
    src_rows = get_src_rows_for_row(row)
    sess_start = run_start_idx[row]
    for k, src in enumerate(src_rows):
        off = k * lag_w
        same_sess = (session_ids[src] == session_ids[row])
        if not same_sess:
            src = sess_start  # clamp to session start for position; zero vel+force
        # tool features
        X_lags[row, off]   = tool_x_all[src]
        X_lags[row, off+1] = tool_y_all[src]
        X_lags[row, off+2] = tool_z_all[src]
        X_lags[row, off+3] = tool_vx_all[src] if same_sess else 0.0
        X_lags[row, off+4] = tool_vy_all[src] if same_sess else 0.0
        X_lags[row, off+5] = tool_vz_all[src] if same_sess else 0.0
        X_lags[row, off+6] = log1p_signed(tool_fx_all[src]) if same_sess else 0.0
        X_lags[row, off+7] = log1p_signed(tool_fy_all[src]) if same_sess else 0.0
        X_lags[row, off+8] = log1p_signed(tool_fz_all[src]) if same_sess else 0.0
        # neighbour features
        nb_off = off + 9
        for j, nv in enumerate(nb):
            feat = [ddx_all[src,nv], ddy_all[src,nv], ddz_all[src,nv]] if same_sess else [0.,0.,0.]
            if USE_ACCSTRESS:
                feat += ([sax_all[src,nv], say_all[src,nv], saz_all[src,nv]] if same_sess else [0.,0.,0.])
            if USE_STRAIN:
                feat += ([rexx_all[src,nv], reyy_all[src,nv], rezz_all[src,nv]] if same_sess else [0.,0.,0.])
            if USE_VEL:
                feat += ([vvx_all[src,nv], vvy_all[src,nv], vvz_all[src,nv]] if same_sess else [0.,0.,0.])
            if USE_STRESS:
                feat += ([rsxx_all[src,nv], rsyy_all[src,nv], rszz_all[src,nv]] if same_sess else [0.,0.,0.])
            if ADD_NORMALS:
                feat += [vertex_normals[nv,0], vertex_normals[nv,1], vertex_normals[nv,2]]
            X_lags[row, nb_off + j*n_nb_feat : nb_off + (j+1)*n_nb_feat] = feat
print(f"Lag features shape: {X_lags.shape}")

# ── Global features ───────────────────────────────────────────────────────────
# dt_cum: time from oldest step to each intermediate step
dt_cum = np.zeros((N_rows, N_STEPS), dtype=np.float32)
for row in range(N_rows):
    src_rows = get_src_rows_for_row(row)
    base_rt = real_time[src_rows[-1]]  # oldest step is the base
    for k, src in enumerate(src_rows):
        if session_ids[src] != session_ids[row]:
            dt_cum[row, k] = 0.0
        else:
            dt_cum[row, k] = np.clip(real_time[src] - base_rt, 0.0, 5.0)

dt_pred = np.zeros((N_rows, 1), dtype=np.float32)
for row in range(N_rows):
    src_rows = get_src_rows_for_row(row)
    base_rt = real_time[src_rows[-1]]
    dt_pred[row, 0] = np.clip(real_time[row] - base_rt, 0.0, 5.0)

current_pos_vel = np.stack([
    tool_x_all, tool_y_all, tool_z_all,
    tool_vx_all, tool_vy_all, tool_vz_all,
], axis=1).astype(np.float32)

delta_min_dist = np.zeros(N_rows, dtype=np.float32)
delta_min_dist[1:] = min_dist_to_mesh[1:] - min_dist_to_mesh[:-1]
force_mag_all = np.linalg.norm(df[['tool_fx','tool_fy','tool_fz']].values, axis=1).astype(np.float32)
lag1_contact = np.zeros(N_rows, dtype=np.float32)
lag1_contact[1:] = (force_mag_all[:-1] > 0.01).astype(np.float32)
contact_features = np.stack([min_dist_to_mesh, delta_min_dist, lag1_contact], axis=1).astype(np.float32)

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

global_parts = [dt_cum, dt_pred, current_pos_vel, contact_features, accel]

if SESSION_PROGRESS:
    sess_progress = np.zeros(N_rows, dtype=np.float32)
    for sid in np.unique(session_ids):
        idx = np.where(session_ids == sid)[0]
        n = len(idx)
        sess_progress[idx] = np.arange(n, dtype=np.float32) / max(n - 1, 1)
    global_parts.append(sess_progress.reshape(-1, 1))

if USE_YOUNGS:
    if 'youngs_modulus' not in df.columns:
        raise ValueError("--use-youngs requires a CSV with 'youngs_modulus' column. "
                         "Run run_E_sweep_collect.sh first, then train on training_data_E_sweep.csv.")
    E_raw = df['youngs_modulus'].values.astype(np.float32)
    # Log-normalise: log(E/1000) maps [500,5000] → [-0.69, 1.61]
    E_norm = np.log(E_raw / 1000.0).reshape(-1, 1)
    global_parts.append(E_norm)
    print(f"youngs_modulus feature: range [{E_raw.min():.0f}, {E_raw.max():.0f}] Pa, "
          f"log-norm range [{E_norm.min():.2f}, {E_norm.max():.2f}]")

X_global = np.concatenate(global_parts, axis=1).astype(np.float32)
assert X_global.shape[1] == n_global, f"Expected {n_global} global features, got {X_global.shape[1]}"

X_combined = np.concatenate([X_global, X_lags], axis=1)
N_INPUTS = X_combined.shape[1]
print(f"Total inputs: {N_INPUTS}")

# ── Variable-window: ALL rows in past VAR_WINDOW seconds ──────────────────────
if VAR_WINDOW > 0:
    # n_global_base: global features WITHOUT dt_cum (dt per step goes into sequence instead)
    # Layout: [dt_pred(1), pos_vel(6), contact(3), accel(3), optional_E, optional_sp]
    n_global_base_vw = 13 + int(SESSION_PROGRESS) + int(USE_YOUNGS)
    T = MAX_STEPS_VAR
    X_var   = np.zeros((N_rows, n_global_base_vw + T + T * lag_w), dtype=np.float32)
    mask_var = np.ones((N_rows, T), dtype=bool)   # True = padded (ignore)

    # Build session time index for fast window lookup
    vw_session_map = {}
    for sid in np.unique(session_ids):
        rows_s = np.where(session_ids == sid)[0]
        vw_session_map[sid] = (rows_s, sim_time[rows_s])

    print(f"Building variable-window features (max_steps={T}, window={VAR_WINDOW}s)...")
    log1p_s = lambda v: np.sign(v) * np.log1p(np.abs(v))
    for row in range(N_rows):
        if row % 20000 == 0:
            print(f"  {row}/{N_rows}")
        sid = session_ids[row]
        rows_s, sts_s = vw_session_map[sid]
        t_now = sim_time[row]
        in_win = (sts_s < t_now) & ((t_now - sts_s) <= VAR_WINDOW)
        src_rows_win = rows_s[in_win]   # ascending time order, excludes current row

        nb = neighbour_idx[row]

        # Fill from most-recent (rank=0) to oldest (rank=n_real-1)
        for rank, src in enumerate(reversed(list(src_rows_win[-T:]))):
            dt_i = t_now - sim_time[src]
            X_var[row, n_global_base_vw + rank] = dt_i          # dt_seq slot
            mask_var[row, rank] = False                          # real position

            off = n_global_base_vw + T + rank * lag_w
            X_var[row, off]   = tool_x_all[src]
            X_var[row, off+1] = tool_y_all[src]
            X_var[row, off+2] = tool_z_all[src]
            X_var[row, off+3] = tool_vx_all[src]
            X_var[row, off+4] = tool_vy_all[src]
            X_var[row, off+5] = tool_vz_all[src]
            X_var[row, off+6] = log1p_s(tool_fx_all[src])
            X_var[row, off+7] = log1p_s(tool_fy_all[src])
            X_var[row, off+8] = log1p_s(tool_fz_all[src])
            nb_off = off + 9
            for j_idx, nv in enumerate(nb):
                feat = [ddx_all[src,nv], ddy_all[src,nv], ddz_all[src,nv]]
                if USE_ACCSTRESS: feat += [sax_all[src,nv], say_all[src,nv], saz_all[src,nv]]
                if USE_STRAIN:    feat += [rexx_all[src,nv], reyy_all[src,nv], rezz_all[src,nv]]
                X_var[row, nb_off + j_idx*n_nb_feat : nb_off + (j_idx+1)*n_nb_feat] = feat

        # Global features (no dt_cum — dt is in sequence)
        # [dt_pred, pos_vel(6), contact(3), accel(3), optional_E, optional_sp]
        base_rt = sim_time[src_rows_win[0]] if len(src_rows_win) > 0 else t_now
        X_var[row, 0]  = np.clip(t_now - base_rt, 0, 5.0)      # dt_pred
        X_var[row, 1]  = tool_x_all[row]
        X_var[row, 2]  = tool_y_all[row]
        X_var[row, 3]  = tool_z_all[row]
        X_var[row, 4]  = tool_vx_all[row]
        X_var[row, 5]  = tool_vy_all[row]
        X_var[row, 6]  = tool_vz_all[row]
        X_var[row, 7]  = min_dist_to_mesh[row]
        X_var[row, 8]  = delta_min_dist[row]
        X_var[row, 9]  = lag1_contact[row]
        X_var[row, 10] = accel[row, 0]
        X_var[row, 11] = accel[row, 1]
        X_var[row, 12] = accel[row, 2]
        gidx = 13
        if SESSION_PROGRESS:
            X_var[row, gidx] = sess_progress[row]; gidx += 1
        if USE_YOUNGS:
            X_var[row, gidx] = E_norm[row, 0]

    print(f"Variable-window features built: X_var={X_var.shape}  mask_var={mask_var.shape}")
    real_per_row = (~mask_var).sum(axis=1)
    print(f"  Real rows per window: min={real_per_row.min()}  mean={real_per_row.mean():.1f}  max={real_per_row.max()}")
    # Replace X_combined with var-window features for downstream code
    X_combined    = X_var
    N_INPUTS      = X_combined.shape[1]
    n_global_base_saved = n_global_base_vw  # saved for model init

# ── Targets ───────────────────────────────────────────────────────────────────
active_verts = [i for i in range(N_VERTICES) if i not in FIXED_INDICES]
dy_cols, pdy_cols = [], []
for i in active_verts:
    dy_cols  += [f'dx{i}', f'dy{i}', f'dz{i}']
    pdy_cols += [f'pdx{i}', f'pdy{i}', f'pdz{i}']
Y_deform = (df[dy_cols].values - df[pdy_cols].values).astype(np.float32)
force_raw = df[['tool_fx','tool_fy','tool_fz']].values.astype(np.float32)
E_raw_all = df['youngs_modulus'].values.astype(np.float32) if USE_YOUNGS else None
del df; import gc; gc.collect()   # free ~12GB pandas frame — all data now in numpy arrays
Y_force   = np.sign(force_raw) * np.log1p(np.abs(force_raw))
N_FORCE   = 3
if FORCE_ONLY:
    Y_combined = Y_force.astype(np.float32)
    print("force-only mode: predicting 3 force outputs only")
else:
    Y_combined = np.concatenate([Y_force, Y_deform], axis=1).astype(np.float32)
N_OUT = Y_combined.shape[1]
print(f"Total outputs: {N_OUT}")

# ── Session-level train/val split (row-balanced) ──────────────────────────────
# Shuffle sessions then greedily assign to val until ~(1-TRAIN_SPLIT) of rows
# are in val — avoids the case where a few large sessions dominate val.
rng = np.random.RandomState(SEED)
all_sids      = np.unique(session_ids)
contact_sids  = all_sids[all_sids != 0]
rng.shuffle(contact_sids)
sid_sizes     = {sid: int((session_ids == sid).sum()) for sid in contact_sids}
total_contact = sum(sid_sizes.values())
target_val    = (1 - TRAIN_SPLIT) * total_contact
# Skip sessions larger than target_val — adding one would massively overshoot.
# Such sessions always go to train; greedy fill runs on the rest.
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

X_train_raw = X_t[train_idx];  X_test_raw = X_t[test_idx]
Y_train_raw = Y_t[train_idx];  Y_test_raw = Y_t[test_idx]
S_train_raw = S_t[train_idx];  S_test_raw = S_t[test_idx]

if VAR_WINDOW > 0:
    mask_t      = torch.BoolTensor(mask_var)
    mask_train  = mask_t[train_idx];  mask_test = mask_t[test_idx]
    perm_mask   = mask_train  # will be reordered below with perm

perm = torch.randperm(len(X_train_raw))
X_train_raw = X_train_raw[perm]; Y_train_raw = Y_train_raw[perm]; S_train_raw = S_train_raw[perm]
if VAR_WINDOW > 0:
    mask_train = mask_train[perm]
print(f"Train: {len(X_train_raw)}  |  Val: {len(X_test_raw)}")

# Normalise
Y_mean = Y_train_raw.mean(0); Y_std = Y_train_raw.std(0) + 1e-8
Y_train = (Y_train_raw - Y_mean) / Y_std; Y_test  = (Y_test_raw  - Y_mean) / Y_std

# Clamp velocity/accel columns before z-scoring X to remove outliers.
if VAR_WINDOW > 0:
    # var-window global layout: [dt_pred(0), pos_vel(1-6), contact(7-9), accel(10-12), ...]
    # tool vx/vy/vz at 4,5,6 — accel at 10,11,12 — per-lag vel at n_global_base_vw+T+k*lag_w+3
    vel_cols = [4, 5, 6]
    for k in range(MAX_STEPS_VAR):
        base = n_global_base_saved + MAX_STEPS_VAR + k * lag_w
        vel_cols += [base+3, base+4, base+5]
    vel_cols += [10, 11, 12]
else:
    # fixed-lag global layout: [dt_cum(N_STEPS), dt_pred, pos_vel(6), contact(3), accel(3)]
    vel_cols = list(range(N_STEPS+4, N_STEPS+7))
    for k in range(N_STEPS):
        base = n_global + k * lag_w
        vel_cols += [base+3, base+4, base+5]
    vel_cols += list(range(n_global-3, n_global))
vel_idx_t = torch.tensor(vel_cols)
q_low  = X_train_raw[:, vel_idx_t].quantile(0.01, dim=0)
q_high = X_train_raw[:, vel_idx_t].quantile(0.99, dim=0)
X_train_raw[:, vel_idx_t] = torch.clamp(X_train_raw[:, vel_idx_t], q_low, q_high)
X_test_raw[:,  vel_idx_t] = torch.clamp(X_test_raw[:,  vel_idx_t], q_low, q_high)

X_mean = X_train_raw.mean(0); X_std = X_train_raw.std(0) + 1e-8
X_train = (X_train_raw - X_mean) / X_std; X_test  = (X_test_raw  - X_mean) / X_std

# ── Sanity: no NaN or Inf anywhere before training ────────────────────────────
for name, t in [('X_train', X_train), ('X_test', X_test),
                ('Y_train', Y_train), ('Y_test',  Y_test)]:
    n_nan = torch.isnan(t).sum().item()
    n_inf = torch.isinf(t).sum().item()
    absmax = t.abs().max().item()
    print(f"  {name}: shape={list(t.shape)}  nan={n_nan}  inf={n_inf}  absmax={absmax:.3f}")
    assert n_nan == 0, f"NaN detected in {name}!"
    assert n_inf == 0, f"Inf detected in {name}!"
    assert absmax < 1e6, f"Extreme values in {name} (absmax={absmax:.1f}) — check features!"
print("SANITY PASSED — no NaN/Inf, values bounded\n")

# ── Model ─────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if VAR_WINDOW > 0:
    model = LiverDualAttnVarLen(N_OUT, MAX_STEPS_VAR, N_NEIGHBOURS, n_nb_feat, n_global_base_saved).to(device)
elif ARCH == 'seq_attn':
    model = LiverSeqAttnFlex(N_OUT, N_STEPS, N_NEIGHBOURS, n_nb_feat).to(device)
elif ARCH == 'dual_attn':
    model = LiverDualAttnFlex(N_OUT, N_STEPS, N_NEIGHBOURS, n_nb_feat).to(device)
else:
    raise ValueError(f"Unknown arch: {ARCH}")

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2, eta_min=1e-6)

MODEL_BEST = f'{SOFA_ROOT}/pinn_project/train/liver_{RUN_ID}_best.pth'
if VAR_WINDOW > 0:
    # Keep data on CPU — input dim is huge (38k+), putting all rows on GPU would OOM.
    # Batches are moved to device inside the training loop instead.
    dataset = TensorDataset(X_train, mask_train, Y_train, S_train_raw)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
else:
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
    for batch in loader:
        if VAR_WINDOW > 0:
            X_b, mask_b, Y_b, S_b = batch
            X_b, mask_b, Y_b, S_b = X_b.to(device), mask_b.to(device), Y_b.to(device), S_b.to(device)
            pred = model(X_b, mask_b)
        else:
            X_b, Y_b, S_b = batch
            pred = model(X_b)
        pred_f = pred[:, :N_FORCE] * y_std_f + y_mean_f
        true_f = Y_b[:, :N_FORCE]  * y_std_f + y_mean_f
        f_mag  = torch.norm(true_f, dim=1, keepdim=True).detach()
        f_w    = 1.0 + (f_mag / (f_mag.mean() + 1e-8)) ** HIGH_FORCE_GAMMA
        s_w    = 1.0 + SUSTAIN_BETA * S_b.unsqueeze(1)
        L_force  = (f_w * s_w * ((pred[:,:N_FORCE] - Y_b[:,:N_FORCE])**2).mean(1, keepdim=True)).mean()
        if FORCE_ONLY:
            L_deform = torch.tensor(0.0, device=device)
            L = L_force
        else:
            L_deform = ((pred[:,N_FORCE:] - Y_b[:,N_FORCE:]) ** 2).mean()
            L = W_FORCE * L_force + W_DEFORM * L_deform
        optimizer.zero_grad(); L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_total += L.item(); epoch_force += L_force.item(); epoch_deform += L_deform.item()

    n_b = len(loader); scheduler.step(epoch)

    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            if VAR_WINDOW > 0:
                val_pred = model(X_test.to(device), mask_test.to(device))
            else:
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
    true = Y_test  * Y_std + Y_mean

    fp = inv_log1p_signed(pred[:, :N_FORCE])
    ft = inv_log1p_signed(true[:, :N_FORCE])
    per_err = torch.norm(fp - ft, dim=1)
    force_rel_err = torch.norm(fp - ft) / (torch.norm(ft) + 1e-8)
    k_out = max(1, len(per_err)//100)
    thresh = torch.topk(per_err, k_out).values.min()
    mask_r = per_err < thresh
    force_rel_robust = torch.norm(fp[mask_r]-ft[mask_r]) / (torch.norm(ft[mask_r])+1e-8)

    sust_m = S_test_raw.bool()
    force_sust = torch.norm(fp[sust_m]-ft[sust_m])/(torch.norm(ft[sust_m])+1e-8) if sust_m.sum() > 0 else torch.tensor(float('nan'))

    if FORCE_ONLY:
        deform_rel = torch.tensor(float('nan'))
        max_mm  = float('nan')
        mean_mm = float('nan')
    else:
        dp = pred[:, N_FORCE:]; dt_out = true[:, N_FORCE:]
        deform_rel = torch.norm(dp - dt_out) / (torch.norm(dt_out) + 1e-8)
        N_ACTIVE = N_VERTICES - len(FIXED_INDICES)
        diff = (dp - dt_out).reshape(-1, N_ACTIVE, 3)
        per_vert = torch.norm(diff, dim=2)
        max_mm  = per_vert.max().item() * 1000
        mean_mm = per_vert.mean().item() * 1000

print(f"\n=== {RUN_ID} RESULTS ===")
print(f"force_rel_l2={force_rel_err.item()*100:.2f}%  "
      f"force_robust={force_rel_robust.item()*100:.2f}%  "
      f"force_sust={force_sust.item()*100:.2f}%")
print(f"deform_rel_l2={deform_rel.item()*100:.2f}%  "
      f"max_vert_err={max_mm:.3f}mm  mean_vert_err={mean_mm:.4f}mm")
print(f"MAE force={per_err.mean().item():.4f}N  RMSE={(((fp-ft)**2).mean().sqrt()).item():.4f}N")

# ── Per-E-value breakdown (only when USE_YOUNGS) ──────────────────────────────
if USE_YOUNGS and E_raw_all is not None:
    E_test_vals  = E_raw_all[test_idx.numpy()]
    E_train_vals = E_raw_all[train_idx.numpy()]
    fm_test_vals = torch.norm(ft, dim=1).numpy()

    with torch.no_grad():
        BATCH = 2048
        trn_preds = []
        for i in range(0, len(X_train), BATCH):
            trn_preds.append(inv_log1p_signed(
                model(X_train[i:i+BATCH].to(device)).cpu() * Y_std + Y_mean
            ))
        fp_train = torch.cat(trn_preds, 0)
        ft_train = inv_log1p_signed(Y_train * Y_std + Y_mean)

    print(f"\n=== PER-E BREAKDOWN ===")
    print(f"{'E(Pa)':>7}  {'val_rows':>9}  {'val_rel_l2':>12}  {'trn_rel_l2':>12}  "
          f"{'mean|F_true|':>14}  {'bias(N)':>9}")
    print("-"*75)
    for e in sorted(np.unique(E_test_vals)):
        vm = torch.BoolTensor(E_test_vals  == e)
        tm = torch.BoolTensor(E_train_vals == e)
        ev = torch.norm(fp[vm]-ft[vm]) / (torch.norm(ft[vm])+1e-8) * 100
        et = torch.norm(fp_train[tm]-ft_train[tm]) / (torch.norm(ft_train[tm])+1e-8) * 100 if tm.sum()>0 else float('nan')
        mft  = torch.norm(ft[vm], dim=1).mean().item()
        bias = (torch.norm(fp[vm], dim=1) - torch.norm(ft[vm], dim=1)).mean().item()
        print(f"  E={int(e):5d}  {vm.sum().item():9d}  {ev.item():11.2f}%  {et.item():11.2f}%  "
              f"{mft:14.4f}N  {bias:+9.4f}N")

# Append summary to shared results file
with open(RESULTS_FILE, 'a') as f:
    f.write(
        f"run={RUN_ID:<25s}  arch={ARCH:<10s}  n_steps={N_STEPS:2d}  K={N_NEIGHBOURS:2d}  "
        f"n_nb_feat={n_nb_feat:2d}  time_w={TIME_WINDOW:.1f}  "
        f"accstress={int(USE_ACCSTRESS)}  strain={int(USE_STRAIN)}  vel={int(USE_VEL)}  stress={int(USE_STRESS)}  "
        f"force_only={int(FORCE_ONLY)}  gamma={HIGH_FORCE_GAMMA:.1f}  sess_prog={int(SESSION_PROGRESS)}  normals={int(ADD_NORMALS)}  "
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
    'use_vel': USE_VEL, 'use_stress': USE_STRESS,
    'time_window': TIME_WINDOW,
    'var_window': VAR_WINDOW, 'max_steps_var': MAX_STEPS_VAR,
    'n_global_base': n_global_base_saved if VAR_WINDOW > 0 else None,
    'force_rel_err': force_rel_err.item(),
    'deform_rel_err': deform_rel.item(),
}, f'{SOFA_ROOT}/pinn_project/train/liver_{RUN_ID}.pth')
print(f"Full checkpoint saved to liver_{RUN_ID}.pth")

# Plot
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(history['epoch'], history['total'],  label='total')
axs[0].plot(history['epoch'], history['force'],  label='force')
axs[0].plot(history['epoch'], history['deform'], label='deform')
axs[0].set_yscale('log'); axs[0].set_xlabel('epoch'); axs[0].legend()
axs[0].set_title(f'{RUN_ID} — training loss')
axs[1].plot(ft[:200, 1].numpy(), label='true fy', alpha=0.7)
axs[1].plot(fp[:200, 1].numpy(), label='pred fy', alpha=0.7)
axs[1].legend(); axs[1].set_title('Force fy — first 200 val samples')
plt.tight_layout()
plt.savefig(f'{SOFA_ROOT}/pinn_project/train/liver_{RUN_ID}_loss.png', dpi=150)
print("Plot saved.")
