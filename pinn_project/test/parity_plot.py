"""
Predict forces on a recorded haptic session and plot FEM vs PINN comparison.

Usage:
  python3 predict_session.py                        # last session in training CSV
  python3 predict_session.py path/to/session.csv    # specific CSV file
  python3 predict_session.py --session-id 42        # specific session from training CSV

Output:
  force_comparison.png — 4-panel plot: fx / fy / fz / |F| vs time
"""

import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'train'))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# ── PARSE ARGS ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('csv', nargs='?', default=None, help='CSV file path')
parser.add_argument('--session-id', type=int, default=None)
parser.add_argument('--fem-skip',   type=int, default=1,
                    help='Simulate FEM latency: use PINN deform every N steps (1=ideal)')
args = parser.parse_args()

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt   = torch.load(MODEL_PATH, map_location=device, weights_only=False)
X_mean = ckpt['X_mean'].cpu().numpy() if hasattr(ckpt['X_mean'], 'numpy') else np.array(ckpt['X_mean'])
X_std  = ckpt['X_std'].cpu().numpy()  if hasattr(ckpt['X_std'],  'numpy') else np.array(ckpt['X_std'])
Y_mean = ckpt['Y_mean'].cpu().numpy() if hasattr(ckpt['Y_mean'], 'numpy') else np.array(ckpt['Y_mean'])
Y_std  = ckpt['Y_std'].cpu().numpy()  if hasattr(ckpt['Y_std'],  'numpy') else np.array(ckpt['Y_std'])
n_in   = ckpt['n_inputs']
n_out  = ckpt['n_output']
n_force= ckpt['n_force']
model  = LiverUNet(n_output=n_out, n_inputs=n_in).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"  Model: {n_in} → {n_out} | Device: {device}")

active_verts = [i for i in range(N_VERTICES) if i not in FIXED_INDICES]

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
csv_path = args.csv or CSV_PATH
print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

# Filter to requested session
if args.session_id is not None:
    df = df[df['session_id'] == args.session_id].reset_index(drop=True)
    if len(df) == 0:
        print(f"ERROR: session_id {args.session_id} not found.")
        sys.exit(1)
    print(f"  Using session {args.session_id}: {len(df)} rows")
elif 'session_id' in df.columns:
    # Use last session with enough data
    valid = df[df['session_id'] > 0]
    if len(valid) > 0:
        best = valid.groupby('session_id').size()
        best = best[best >= 20]
        if len(best) > 0:
            last_sid = best.index[-1]
            df = df[df['session_id'] == last_sid].reset_index(drop=True)
            print(f"  Auto-selected session {last_sid}: {len(df)} rows")
        else:
            print(f"  Using all {len(df)} rows (no session filter)")
    else:
        print(f"  Using all {len(df)} rows")
else:
    print(f"  Using all {len(df)} rows")

N_rows = len(df)
if N_rows < 10:
    print("ERROR: Not enough rows in session."); sys.exit(1)

# ── RAW ARRAYS ────────────────────────────────────────────────────────────────
dx_all  = df[[f'dx{i}'   for i in range(N_VERTICES)]].values.astype(np.float32)
dy_all  = df[[f'dy{i}'   for i in range(N_VERTICES)]].values.astype(np.float32)
dz_all  = df[[f'dz{i}'   for i in range(N_VERTICES)]].values.astype(np.float32)
pdx_all = df[[f'pdx{i}'  for i in range(N_VERTICES)]].values.astype(np.float32)
pdy_all = df[[f'pdy{i}'  for i in range(N_VERTICES)]].values.astype(np.float32)
pdz_all = df[[f'pdz{i}'  for i in range(N_VERTICES)]].values.astype(np.float32)
ddx_all = dx_all - pdx_all
ddy_all = dy_all - pdy_all
ddz_all = dz_all - pdz_all
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
sess    = df['session_id'].values if 'session_id' in df.columns else np.zeros(N_rows)

true_force = np.stack([tool_fx, tool_fy, tool_fz], axis=1)  # (N,3)

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

# ── HELPERS ───────────────────────────────────────────────────────────────────
def decode_deform(pred_534):
    ox, oy, oz = np.zeros(N_VERTICES), np.zeros(N_VERTICES), np.zeros(N_VERTICES)
    for k, vi in enumerate(active_verts):
        ox[vi] = pred_534[k*3+0]; oy[vi] = pred_534[k*3+1]; oz[vi] = pred_534[k*3+2]
    return ox, oy, oz

def build_X(row, buf_ddx, buf_ddy, buf_ddz, buf_sax, buf_say, buf_saz,
            buf_rxx, buf_ryy, buf_rzz):
    nb = nb_idx[row]
    base_r  = max(0, row - N_LAGS)
    rt_base = rt_vals[base_r]
    dt_cum  = np.zeros(N_LAGS, dtype=np.float32)
    for lag in range(1, N_LAGS + 1):
        src = max(0, row - lag)
        if sess[src] != sess[row]: dt_cum[lag-1] = 0.0
        else: dt_cum[lag-1] = float(np.clip(rt_vals[src] - rt_base, 0, 0.15))
    dt_pred = float(np.clip(rt_vals[row] - rt_base, 0, 0.15))
    cur = np.array([tool_x[row], tool_y[row], tool_z[row],
                    tool_vx[row], tool_vy[row], tool_vz[row]], dtype=np.float32)
    d_dist = 0.0 if row == 0 or sess[row-1] != sess[row] else float(min_dist_arr[row] - min_dist_arr[row-1])
    prev_f = float(np.linalg.norm([tool_fx[row-1], tool_fy[row-1], tool_fz[row-1]])) if row > 0 else 0.0
    cf = np.array([min_dist_arr[row], d_dist, float(prev_f > 0.01)], dtype=np.float32)
    th = []
    for lag in range(1, N_LAGS + 1):
        src = max(0, row - lag)
        if sess[src] != sess[row]:
            ss0 = int(np.where(sess == sess[row])[0][0])
            th.extend([tool_x[ss0], tool_y[ss0], tool_z[ss0], 0, 0, 0, 0, 0, 0])
        else:
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

# ── RUN PINN ──────────────────────────────────────────────────────────────────
print(f"Running PINN prediction (FEM_SKIP={args.fem_skip})...")
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
    X      = build_X(row, buf_ddx, buf_ddy, buf_ddz, buf_sax, buf_say, buf_saz, buf_rxx, buf_ryy, buf_rzz)
    X_norm = (X - X_mean) / (X_std + 1e-8)
    X_t    = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        Y_norm = model(X_t).cpu().numpy()[0]
    Y_pred = Y_norm * Y_std + Y_mean
    fp_log = Y_pred[:n_force]
    pinn_force[row] = np.sign(fp_log) * np.expm1(np.abs(fp_log))

    fem_avail = (row % args.fem_skip == 0)
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

# ── METRICS ───────────────────────────────────────────────────────────────────
true_mag  = np.linalg.norm(true_force, axis=1)
pinn_mag  = np.linalg.norm(pinn_force, axis=1)
err_mag   = np.abs(pinn_mag - true_mag)
rel_l2    = 100 * np.linalg.norm(pinn_force - true_force) / (np.linalg.norm(true_force) + 1e-8)
mae       = float(np.linalg.norm(pinn_force - true_force, axis=1).mean())
median_e  = float(np.median(np.linalg.norm(pinn_force - true_force, axis=1)))

print(f"\n=== Results (FEM_SKIP={args.fem_skip}) ===")
print(f"  Rel L2:   {rel_l2:.2f}%")
print(f"  MAE:      {mae:.4f} N")
print(f"  Median:   {median_e:.4f} N")
print(f"  True |F| mean: {true_mag.mean():.3f} N  max: {true_mag.max():.3f} N")
print(f"  PINN |F| mean: {pinn_mag.mean():.3f} N  max: {pinn_mag.max():.3f} N")


# ── PARITY PLOT: predicted vs true, one point per state, no time/position/session ──
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
fig.suptitle(f'PINN Parity Plot — predicted vs ground-truth force, per state  '
             f'(FEM_SKIP={args.fem_skip}, n={N_rows} states)\n'
             f'Rel L2: {rel_l2:.1f}%  MAE: {mae:.3f}N  |  perfect prediction = diagonal line',
             fontsize=12, fontweight='bold')

labels   = ['Fx (N)', 'Fy (N)', 'Fz (N)', '|F| (N)']
true_set = [true_force[:, 0], true_force[:, 1], true_force[:, 2], true_mag]
pred_set = [pinn_force[:, 0], pinn_force[:, 1], pinn_force[:, 2], pinn_mag]

for i, ax in enumerate(axes):
    t_vals, p_vals = true_set[i], pred_set[i]
    lo, hi = min(t_vals.min(), p_vals.min()), max(t_vals.max(), p_vals.max())
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.6, label='perfect (y=x)')
    ax.scatter(t_vals, p_vals, s=10, alpha=0.4, color='tomato')
    ax.set_xlabel(f'True {labels[i]}', fontsize=10)
    ax.set_ylabel(f'Predicted {labels[i]}', fontsize=10)
    ax.set_title(labels[i], fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    if i == 0:
        ax.legend(loc='upper left', fontsize=8)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath('parity_plot.py')), f'parity_plot_skip{args.fem_skip}.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nParity plot saved -> {out_path}")
