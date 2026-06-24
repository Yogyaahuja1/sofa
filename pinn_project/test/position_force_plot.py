"""
PINN ON vs OFF, force plotted against POSITION (cumulative distance travelled
along the path), not time. Two panels: FEM_SKIP=1 (ideal) and FEM_SKIP=3
(realistic intermittent FEM).

Raw x/y/z coordinates are NOT monotonic along this path (tool moves back and
forth in all 3 axes), so plotting against a single raw coordinate would
zigzag and be unreadable. Cumulative arc-length distance is monotonic by
construction and still genuinely represents "position along the trajectory."

Usage:
  python3 position_force_plot.py --session-id 35
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
parser.add_argument('--session-id', type=int, default=None,
                    help='Restrict to one touch episode. Omit to use the WHOLE recorded path.')
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

df = pd.read_csv(CSV_PATH).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
if args.session_id is not None:
    print(f"Loading session {args.session_id}...")
    df = df[df['session_id'] == args.session_id].reset_index(drop=True)
else:
    print("Loading recorded path, excluding episodes <15 rows (never part of training)...")
    sess_counts = df.groupby('session_id')['session_id'].transform('count')
    df = df[(df['session_id'] == 0) | (sess_counts >= N_LAGS * 3)].reset_index(drop=True)
N_rows = len(df)
print(f"  {N_rows} rows")

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
sess    = df['session_id'].values

true_force = np.stack([tool_fx, tool_fy, tool_fz], axis=1)

# Position axis: cumulative distance travelled along the path (monotonic, unlike raw x/y/z)
pos_xyz = np.stack([tool_x, tool_y, tool_z], axis=1)
step_dist = np.zeros(N_rows, dtype=np.float32)
step_dist[1:] = np.linalg.norm(np.diff(pos_xyz, axis=0), axis=1)
position = np.cumsum(step_dist)

print("Computing neighbour indices...")
vertex_pos   = np.load(VERTICES_PATH)
tool_pos_all = pos_xyz
nb_idx       = np.zeros((N_rows, N_NEIGHBOURS), dtype=int)
min_dist_arr = np.zeros(N_rows, dtype=np.float32)
for r in range(N_rows):
    d = np.linalg.norm(vertex_pos - tool_pos_all[r], axis=1)
    s = np.argsort(d)
    nb_idx[r] = s[:N_NEIGHBOURS]
    min_dist_arr[r] = d[s[0]]

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

    rel_l2 = 100 * np.linalg.norm(pinn_force - true_force) / (np.linalg.norm(true_force) + 1e-8)
    mae    = float(np.linalg.norm(pinn_force - true_force, axis=1).mean())
    contact_mask = np.linalg.norm(true_force, axis=1) > 0.01
    rel_l2_contact = 100 * np.linalg.norm(pinn_force[contact_mask] - true_force[contact_mask]) / \
                     (np.linalg.norm(true_force[contact_mask]) + 1e-8)
    print(f"  MAE: {mae:.4f} N   Rel L2 (contact only): {rel_l2_contact:.2f}%")
    # per-session error breakdown — which episodes are actually bad?
    per_sess_err = np.linalg.norm(pinn_force - true_force, axis=1)
    tmp = pd.DataFrame({'session_id': sess, 'err': per_sess_err, 'true_mag': np.linalg.norm(true_force, axis=1)})
    sess_summary = tmp[tmp.session_id != 0].groupby('session_id').agg(
        mean_err=('err', 'mean'), mean_true=('true_mag', 'mean'), n=('err', 'count'))
    sess_summary['rel_pct'] = 100 * sess_summary.mean_err / (sess_summary.mean_true + 1e-8)
    print("  Worst 10 episodes by mean error:")
    print(sess_summary.sort_values('mean_err', ascending=False).head(10))
    return pinn_force, rel_l2

print("Running FEM_SKIP=1 (ideal, PINN ON)...")
pinn_skip1, rel_l2_1 = run_pinn(1)
print(f"  Rel L2: {rel_l2_1:.2f}%")

print("Running FEM_SKIP=3 (realistic, PINN ON)...")
pinn_skip3, rel_l2_3 = run_pinn(3)
print(f"  Rel L2: {rel_l2_3:.2f}%")

true_mag = np.linalg.norm(true_force, axis=1)
mag1     = np.linalg.norm(pinn_skip1, axis=1)
mag3     = np.linalg.norm(pinn_skip3, axis=1)

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
path_label = f'session {args.session_id}' if args.session_id is not None else 'recorded path, episodes >=15 rows only'
fig.suptitle(f'PINN ON vs OFF — |Force| vs Position along path ({path_label})',
             fontsize=13, fontweight='bold')

axes[0].plot(position, true_mag, color='royalblue', linewidth=1.3, label='PINN OFF (FEM ground truth)')
axes[0].plot(position, mag1,     color='orangered', linewidth=1.1, linestyle='--', label='PINN ON (FEM_SKIP=1, ideal)')
axes[0].set_ylabel('|F| (N)', fontsize=10)
axes[0].set_title(f'FEM_SKIP=1 (ideal)  |  Rel L2: {rel_l2_1:.1f}%', fontsize=10)
axes[0].legend(loc='upper right', fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(position, true_mag, color='royalblue', linewidth=1.3, label='PINN OFF (FEM ground truth)')
axes[1].plot(position, mag3,     color='orangered', linewidth=1.1, linestyle='--', label='PINN ON (FEM_SKIP=3, realistic)')
axes[1].set_ylabel('|F| (N)', fontsize=10)
axes[1].set_xlabel('Position along path — cumulative distance travelled', fontsize=10)
axes[1].set_title(f'FEM_SKIP=3 (realistic intermittent FEM)  |  Rel L2: {rel_l2_3:.1f}%', fontsize=10)
axes[1].legend(loc='upper right', fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out_suffix = f'_session{args.session_id}' if args.session_id is not None else '_filtered15'
out_path = f'/home/yogyaahuja/sofa/pinn_project/test/position_force_comparison{out_suffix}.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")
