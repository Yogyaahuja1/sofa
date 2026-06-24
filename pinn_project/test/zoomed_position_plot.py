"""
Zoomed-in PINN ON vs OFF comparison over a short (1-2 second) window of real time,
plotted against cumulative distance travelled (position), not time, with individual
sample markers so step-by-step prediction behaviour is visible.

Usage:
  python3 zoomed_position_plot.py --session-id 35 --start-time 60 --duration 2.0
"""

import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'train'))

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pinn_model import LiverUNet

MODEL_PATH    = '/home/yogyaahuja/sofa/pinn_project/train/tissue_pinn_force_final.pth'
CSV_PATH      = '/home/yogyaahuja/sofa/pinn_project/data/training_data.csv'
VERTICES_PATH = '/home/yogyaahuja/sofa/pinn_project/data/liver_vertices.npy'
N_NEIGHBOURS, N_LAGS, N_VERTICES = 20, 5, 181
FIXED_INDICES = [3, 39, 64]
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument('--session-id', type=int, default=35)
parser.add_argument('--start-time', type=float, default=None, help='real_time seconds to start the zoom window (default: auto-pick a contact-rich region)')
parser.add_argument('--duration', type=float, default=2.0, help='zoom window duration in seconds')
parser.add_argument('--fem-skip', type=int, default=3)
args = parser.parse_args()

torch.manual_seed(SEED); np.random.seed(SEED)

print("Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt   = torch.load(MODEL_PATH, map_location=device, weights_only=False)
X_mean = ckpt['X_mean'].cpu().numpy() if hasattr(ckpt['X_mean'], 'numpy') else np.array(ckpt['X_mean'])
X_std  = ckpt['X_std'].cpu().numpy()  if hasattr(ckpt['X_std'],  'numpy') else np.array(ckpt['X_std'])
Y_mean = ckpt['Y_mean'].cpu().numpy() if hasattr(ckpt['Y_mean'], 'numpy') else np.array(ckpt['Y_mean'])
Y_std  = ckpt['Y_std'].cpu().numpy()  if hasattr(ckpt['Y_std'],  'numpy') else np.array(ckpt['Y_std'])
n_in, n_out, n_force = ckpt['n_inputs'], ckpt['n_output'], ckpt['n_force']
model = LiverUNet(n_output=n_out, n_inputs=n_in).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()
active_verts = [i for i in range(N_VERTICES) if i not in FIXED_INDICES]

print(f"Loading session {args.session_id} (needs full session for valid lag history)...")
full_df = pd.read_csv(CSV_PATH).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
full_df = full_df[full_df['session_id'] == args.session_id].sort_values('step').reset_index(drop=True)
N_rows = len(full_df)
print(f"  {N_rows} total rows in session, real_time range "
      f"[{full_df.real_time.min():.2f}, {full_df.real_time.max():.2f}]")

if args.start_time is None:
    # auto-pick the window with the highest mean |force| (most contact-rich)
    fmag_all = np.sqrt(full_df.tool_fx**2 + full_df.tool_fy**2 + full_df.tool_fz**2).values
    best_i, best_mean = 0, -1
    for i in range(N_rows):
        end_t = full_df.real_time.iloc[i] + args.duration
        j = np.searchsorted(full_df.real_time.values, end_t)
        if j <= i + 3 or j > N_rows: continue
        m = fmag_all[i:j].mean()
        if m > best_mean:
            best_mean, best_i = m, i
    args.start_time = float(full_df.real_time.iloc[best_i])
    print(f"  Auto-picked contact-rich window starting at real_time={args.start_time:.2f}s")

dx_all  = full_df[[f'dx{i}'   for i in range(N_VERTICES)]].values.astype(np.float32)
dy_all  = full_df[[f'dy{i}'   for i in range(N_VERTICES)]].values.astype(np.float32)
dz_all  = full_df[[f'dz{i}'   for i in range(N_VERTICES)]].values.astype(np.float32)
pdx_all = full_df[[f'pdx{i}'  for i in range(N_VERTICES)]].values.astype(np.float32)
pdy_all = full_df[[f'pdy{i}'  for i in range(N_VERTICES)]].values.astype(np.float32)
pdz_all = full_df[[f'pdz{i}'  for i in range(N_VERTICES)]].values.astype(np.float32)
ddx_all = dx_all - pdx_all
ddy_all = dy_all - pdy_all
ddz_all = dz_all - pdz_all
sax_all = full_df[[f'sax{i}'  for i in range(N_VERTICES)]].values.astype(np.float32)
say_all = full_df[[f'say{i}'  for i in range(N_VERTICES)]].values.astype(np.float32)
saz_all = full_df[[f'saz{i}'  for i in range(N_VERTICES)]].values.astype(np.float32)
rxx_all = full_df[[f'rexx{i}' for i in range(N_VERTICES)]].values.astype(np.float32)
ryy_all = full_df[[f'reyy{i}' for i in range(N_VERTICES)]].values.astype(np.float32)
rzz_all = full_df[[f'rezz{i}' for i in range(N_VERTICES)]].values.astype(np.float32)

tool_x  = full_df['tool_x'].values.astype(np.float32)
tool_y  = full_df['tool_y'].values.astype(np.float32)
tool_z  = full_df['tool_z'].values.astype(np.float32)
tool_vx = full_df['tool_vx'].values.astype(np.float32)
tool_vy = full_df['tool_vy'].values.astype(np.float32)
tool_vz = full_df['tool_vz'].values.astype(np.float32)
tool_fx = full_df['tool_fx'].values.astype(np.float32)
tool_fy = full_df['tool_fy'].values.astype(np.float32)
tool_fz = full_df['tool_fz'].values.astype(np.float32)
rt_vals = full_df['real_time'].values.astype(np.float32)
sess    = full_df['session_id'].values

true_force = np.stack([tool_fx, tool_fy, tool_fz], axis=1)
pos_xyz = np.stack([tool_x, tool_y, tool_z], axis=1)
step_dist = np.zeros(N_rows, dtype=np.float32)
step_dist[1:] = np.linalg.norm(np.diff(pos_xyz, axis=0), axis=1)
position = np.cumsum(step_dist)

print("Computing neighbour indices...")
vertex_pos = np.load(VERTICES_PATH)
nb_idx = np.zeros((N_rows, N_NEIGHBOURS), dtype=int)
min_dist_arr = np.zeros(N_rows, dtype=np.float32)
for r in range(N_rows):
    d = np.linalg.norm(vertex_pos - pos_xyz[r], axis=1)
    s = np.argsort(d)
    nb_idx[r] = s[:N_NEIGHBOURS]
    min_dist_arr[r] = d[s[0]]

def decode_deform(pred_534):
    ox, oy, oz = np.zeros(N_VERTICES), np.zeros(N_VERTICES), np.zeros(N_VERTICES)
    for k, vi in enumerate(active_verts):
        ox[vi] = pred_534[k*3+0]; oy[vi] = pred_534[k*3+1]; oz[vi] = pred_534[k*3+2]
    return ox, oy, oz

def build_X(row, buf_ddx, buf_ddy, buf_ddz, buf_sax, buf_say, buf_saz, buf_rxx, buf_ryy, buf_rzz):
    nb = nb_idx[row]
    base_r  = max(0, row - N_LAGS)
    rt_base = rt_vals[base_r]
    dt_cum  = np.zeros(N_LAGS, dtype=np.float32)
    for lag in range(1, N_LAGS + 1):
        src = max(0, row - lag)
        dt_cum[lag-1] = float(np.clip(rt_vals[src] - rt_base, 0, 0.15)) if sess[src]==sess[row] else 0.0
    dt_pred = float(np.clip(rt_vals[row] - rt_base, 0, 0.15))
    cur = np.array([tool_x[row], tool_y[row], tool_z[row],
                    tool_vx[row], tool_vy[row], tool_vz[row]], dtype=np.float32)
    d_dist = 0.0 if row == 0 else float(min_dist_arr[row] - min_dist_arr[row-1])
    prev_f = float(np.linalg.norm([tool_fx[row-1], tool_fy[row-1], tool_fz[row-1]])) if row > 0 else 0.0
    cf = np.array([min_dist_arr[row], d_dist, float(prev_f > 0.01)], dtype=np.float32)
    th = []
    for lag in range(1, N_LAGS + 1):
        src = max(0, row - lag)
        th.extend([tool_x[src], tool_y[src], tool_z[src],
                   tool_vx[src], tool_vy[src], tool_vz[src],
                   float(np.sign(tool_fx[src]) * np.log1p(abs(tool_fx[src]))),
                   float(np.sign(tool_fy[src]) * np.log1p(abs(tool_fy[src]))),
                   float(np.sign(tool_fz[src]) * np.log1p(abs(tool_fz[src])))])
    th = np.array(th, dtype=np.float32)
    df_f, sf_f, rf_f = [], [], []
    for lag in range(1, N_LAGS + 1):
        li = lag - 1
        df_f.extend([buf_ddx[li][nb], buf_ddy[li][nb], buf_ddz[li][nb]])
        sf_f.extend([buf_sax[li][nb], buf_say[li][nb], buf_saz[li][nb]])
        rf_f.extend([buf_rxx[li][nb], buf_ryy[li][nb], buf_rzz[li][nb]])
    return np.concatenate([dt_cum, [dt_pred], cur, cf, th,
                           np.concatenate(df_f), np.concatenate(sf_f),
                           np.concatenate(rf_f)]).astype(np.float32)

def run_pinn(fem_skip):
    buf_ddx = [ddx_all[0].copy()] * N_LAGS
    buf_ddy = [ddy_all[0].copy()] * N_LAGS
    buf_ddz = [ddz_all[0].copy()] * N_LAGS
    buf_sax = [sax_all[0].copy()] * N_LAGS
    buf_say = [say_all[0].copy()] * N_LAGS
    buf_saz = [saz_all[0].copy()] * N_LAGS
    buf_rxx = [rxx_all[0].copy()] * N_LAGS
    buf_ryy = [ryy_all[0].copy()] * N_LAGS
    buf_rzz = [rzz_all[0].copy()] * N_LAGS
    last_sax, last_say, last_saz = sax_all[0].copy(), say_all[0].copy(), saz_all[0].copy()
    last_rxx, last_ryy, last_rzz = rxx_all[0].copy(), ryy_all[0].copy(), rzz_all[0].copy()
    pinn_force = np.zeros((N_rows, 3), dtype=np.float32)
    for row in range(N_rows):
        X = build_X(row, buf_ddx, buf_ddy, buf_ddz, buf_sax, buf_say, buf_saz, buf_rxx, buf_ryy, buf_rzz)
        X_norm = (X - X_mean) / (X_std + 1e-8)
        X_t = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            Y_norm = model(X_t).cpu().numpy()[0]
        Y_pred = Y_norm * Y_std + Y_mean
        fp_log = Y_pred[:n_force]
        pinn_force[row] = np.sign(fp_log) * np.expm1(np.abs(fp_log))
        fem_avail = (row % fem_skip == 0)
        if fem_avail:
            new_ddx, new_ddy, new_ddz = ddx_all[row].copy(), ddy_all[row].copy(), ddz_all[row].copy()
            new_sax, last_sax = sax_all[row].copy(), sax_all[row].copy()
            new_say, last_say = say_all[row].copy(), say_all[row].copy()
            new_saz, last_saz = saz_all[row].copy(), saz_all[row].copy()
            new_rxx, last_rxx = rxx_all[row].copy(), rxx_all[row].copy()
            new_ryy, last_ryy = ryy_all[row].copy(), ryy_all[row].copy()
            new_rzz, last_rzz = rzz_all[row].copy(), rzz_all[row].copy()
        else:
            pred_534 = Y_pred[n_force:]
            new_ddx, new_ddy, new_ddz = decode_deform(pred_534)
            new_sax, new_say, new_saz = last_sax, last_say, last_saz
            new_rxx, new_ryy, new_rzz = last_rxx, last_ryy, last_rzz
        buf_ddx = [new_ddx] + buf_ddx[:-1]
        buf_ddy = [new_ddy] + buf_ddy[:-1]
        buf_ddz = [new_ddz] + buf_ddz[:-1]
        buf_sax = [new_sax] + buf_sax[:-1]
        buf_say = [new_say] + buf_say[:-1]
        buf_saz = [new_saz] + buf_saz[:-1]
        buf_rxx = [new_rxx] + buf_rxx[:-1]
        buf_ryy = [new_ryy] + buf_ryy[:-1]
        buf_rzz = [new_rzz] + buf_rzz[:-1]
    return pinn_force

print(f"Running PINN (FEM_SKIP={args.fem_skip}) over full session for valid history, then cropping to zoom window...")
pinn_force = run_pinn(args.fem_skip)

# crop to the zoom window AFTER running predictions on the full session
# (lag history must be built from the full sequence, not just the window)
mask = (rt_vals >= args.start_time) & (rt_vals <= args.start_time + args.duration)
pos_z   = position[mask]
true_z  = true_force[mask]
pinn_z  = pinn_force[mask]
n_pts   = mask.sum()
print(f"Zoom window: real_time [{args.start_time:.2f}, {args.start_time+args.duration:.2f}]s -> {n_pts} points")

true_mag_z = np.linalg.norm(true_z, axis=1)
pinn_mag_z = np.linalg.norm(pinn_z, axis=1)
rel_l2_z = 100 * np.linalg.norm(pinn_z - true_z) / (np.linalg.norm(true_z) + 1e-8)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(pos_z, true_mag_z, color='royalblue', linewidth=1.5, marker='o', markersize=4,
        label='PINN OFF (FEM ground truth)')
ax.plot(pos_z, pinn_mag_z, color='orangered', linewidth=1.3, linestyle='--', marker='x', markersize=5,
        label=f'PINN ON (FEM_SKIP={args.fem_skip})')
ax.set_xlabel('Position along path — cumulative distance travelled', fontsize=10)
ax.set_ylabel('|F| (N)', fontsize=10)
ax.set_title(f'Zoomed: {args.duration}s window @ t={args.start_time:.2f}s, session {args.session_id}  |  '
             f'Rel L2 in window: {rel_l2_z:.1f}%  ({n_pts} points)', fontsize=11, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = '/home/yogyaahuja/sofa/pinn_project/test/zoomed_position_force.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")
