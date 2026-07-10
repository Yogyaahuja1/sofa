"""
The full version of sir's drift-correction idea: not just comparing PINN's
existing (replay-fed) prediction against an interpolated reference force, but
actually REBUILDING the model's input — tool history AND neighbor
deformation/stress/strain — using values interpolated from the real recording
at the replay's ACTUAL position trajectory, then running the model fresh on
this corrected input. This answers "does the model predict well, given inputs
reconstructed to match what really happened at the positions replay actually
reached" — closing the loop that compare_force_interpolated.py left open
(that one only corrected the comparison target, not the model's input).

Requires /tmp/cpp_position_seq.csv (the replay's actual position at every
predictForce call, dumped by PINNPredictor.cpp) — run the replay first.

Usage:
  python3 predict_corrected_replay.py --session-id 85
"""
import argparse
import numpy as np
import pandas as pd
import torch
import sys, os
SOFA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'train'))
from pinn_model import LiverDualAttnFlex

parser = argparse.ArgumentParser()
parser.add_argument('--session-id', type=int, nargs='+', required=True)
args = parser.parse_args()

N_LAGS = 8
N_NB = 20
N_NB_FEAT = 9   # deform(3) + accstress(3) + strain(3) — matches R10
N_V = 181
FIXED = [3, 39, 64]

# ── Load the original recorded path (the interpolation reference) ──────────────
df = pd.read_csv(f'{SOFA_ROOT}/pinn_project/data/training_data.csv')
sess = df[df.session_id.isin(args.session_id)].sort_values(['session_id','step']).reset_index(drop=True)
gaps = sess['step'].diff().abs()
gap_idx = list(gaps[gaps > 50].index)
if gap_idx:
    bounds = [0] + gap_idx + [len(sess)]
    blocks = [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]
    best = max(blocks, key=lambda b: b[1]-b[0])
    sess = sess.iloc[best[0]:best[1]].reset_index(drop=True)
n_ref = len(sess)
print(f"Reference path: {n_ref} recorded rows")

ref_pos   = sess[['tool_x', 'tool_y', 'tool_z']].values
ref_vel   = sess[['tool_vx', 'tool_vy', 'tool_vz']].values
ref_force = sess[['tool_fx', 'tool_fy', 'tool_fz']].values
ref_time  = sess['real_time'].values

ddx_all = (sess[[f'dx{i}' for i in range(N_V)]].values - sess[[f'pdx{i}' for i in range(N_V)]].values)
ddy_all = (sess[[f'dy{i}' for i in range(N_V)]].values - sess[[f'pdy{i}' for i in range(N_V)]].values)
ddz_all = (sess[[f'dz{i}' for i in range(N_V)]].values - sess[[f'pdz{i}' for i in range(N_V)]].values)
sax_all = sess[[f'sax{i}' for i in range(N_V)]].values
say_all = sess[[f'say{i}' for i in range(N_V)]].values
saz_all = sess[[f'saz{i}' for i in range(N_V)]].values
rxx_all = sess[[f'rexx{i}' for i in range(N_V)]].values
ryy_all = sess[[f'reyy{i}' for i in range(N_V)]].values
rzz_all = sess[[f'rezz{i}' for i in range(N_V)]].values

vertex_pos = np.load(f'{SOFA_ROOT}/pinn_project/data/liver_vertices.npy')

# ── Load the replay's actual position at every call ─────────────────────────────
pos_seq = pd.read_csv('/tmp/cpp_position_seq.csv', header=None, names=['call', 'tx', 'ty', 'tz'])
actual_pos = pos_seq[['tx', 'ty', 'tz']].values
n_calls = len(actual_pos)


def find_bracket(query, path):
    best_dist = np.inf
    best_i, best_t = 0, 0.0
    for i in range(len(path) - 1):
        a, b = path[i], path[i+1]
        ab = b - a
        ab_len2 = np.dot(ab, ab)
        t = 0.0 if ab_len2 < 1e-12 else np.clip(np.dot(query - a, ab) / ab_len2, 0.0, 1.0)
        proj = a + t * ab
        dist = np.linalg.norm(query - proj)
        if dist < best_dist:
            best_dist = dist
            best_i, best_t = i, t
    return best_i, best_t, best_dist


def interp_at(query):
    """Return (pos, vel, force, ddx[181], ddy[181], ddz[181], sax.., say.., saz..,
    rxx.., ryy.., rzz..) interpolated at the closest point on the recorded path."""
    i, t, d = find_bracket(query, ref_pos)
    pos = ref_pos[i] + t * (ref_pos[i+1] - ref_pos[i])
    vel = ref_vel[i] + t * (ref_vel[i+1] - ref_vel[i])
    force = ref_force[i] + t * (ref_force[i+1] - ref_force[i])
    rtime = ref_time[i] + t * (ref_time[i+1] - ref_time[i])
    ddx = ddx_all[i] + t * (ddx_all[i+1] - ddx_all[i])
    ddy = ddy_all[i] + t * (ddy_all[i+1] - ddy_all[i])
    ddz = ddz_all[i] + t * (ddz_all[i+1] - ddz_all[i])
    sax = sax_all[i] + t * (sax_all[i+1] - sax_all[i])
    say = say_all[i] + t * (say_all[i+1] - say_all[i])
    saz = saz_all[i] + t * (saz_all[i+1] - saz_all[i])
    rxx = rxx_all[i] + t * (rxx_all[i+1] - rxx_all[i])
    ryy = ryy_all[i] + t * (ryy_all[i+1] - ryy_all[i])
    rzz = rzz_all[i] + t * (rzz_all[i+1] - rzz_all[i])
    return pos, vel, force, rtime, ddx, ddy, ddz, sax, say, saz, rxx, ryy, rzz, d


print("Interpolating corrected reference at every replay step (this is O(n_calls * n_ref), may take a bit)...")
interp_cache = [interp_at(actual_pos[k]) for k in range(n_calls)]
print("Done interpolating.")

# ── Build corrected feature vectors for steps with a full lag window available ──
X_list, true_force_list = [], []
for row in range(N_LAGS, n_calls):
    nb_pos = interp_cache[row][0]
    d = np.linalg.norm(vertex_pos - nb_pos, axis=1)
    nb = np.argsort(d)[:N_NB]
    min_dist = d[nb[0]]

    base_rtime = interp_cache[row - N_LAGS][3]
    # dt_cum: k=0=most recent lag, k=N_LAGS-1=oldest; clip matches training (5.0s)
    dt_cum = np.array([np.clip(interp_cache[row-(k+1)][3] - base_rtime, 0, 5.0) for k in range(N_LAGS)], dtype=np.float32)
    dt_pred = np.clip(interp_cache[row][3] - base_rtime, 0, 5.0)

    cur_pos, cur_vel = interp_cache[row][0], interp_cache[row][1]
    prev_pos = interp_cache[row-1][0] if row >= 1 else cur_pos
    d_dist = min_dist - np.linalg.norm(vertex_pos - prev_pos, axis=1).min()
    prev_force_mag = np.linalg.norm(interp_cache[row-1][2])
    cf = np.array([min_dist, d_dist, float(prev_force_mag > 0.01)], dtype=np.float32)

    # Accel in global section (lag0=most recent vs lag2)
    src1, src3 = interp_cache[row-1], interp_cache[row-3] if row >= 3 else interp_cache[0]
    dt_a = src1[3] - src3[3]
    accel = ((src1[1] - src3[1]) / dt_a) if dt_a > 1e-4 else np.zeros(3)
    accel = np.clip(accel, -500.0, 500.0).astype(np.float32)

    # R10 layout: global(21) + per-lag-interleaved(8 × 189)
    # Per-lag k=0..7 (k=0 = most recent = lag1): tool(9) + [ddx_j,ddy_j,ddz_j,sax_j,say_j,saz_j,rxx_j,ryy_j,rzz_j] × N_NB
    lag_blocks = []
    for k in range(N_LAGS):
        p, v, f, _, ddx, ddy, ddz, sax, say, saz, rxx, ryy, rzz, _ = interp_cache[row - (k + 1)]
        tool_k = [p[0], p[1], p[2], v[0], v[1], v[2],
                  np.sign(f[0])*np.log1p(abs(f[0])), np.sign(f[1])*np.log1p(abs(f[1])), np.sign(f[2])*np.log1p(abs(f[2]))]
        nb_k = []
        for j in nb:
            nb_k.extend([ddx[j], ddy[j], ddz[j], sax[j], say[j], saz[j], rxx[j], ryy[j], rzz[j]])
        lag_blocks.extend(tool_k + nb_k)

    X = np.concatenate([dt_cum, [dt_pred],
                         [cur_pos[0], cur_pos[1], cur_pos[2], cur_vel[0], cur_vel[1], cur_vel[2]],
                         cf, accel,
                         lag_blocks]).astype(np.float32)
    X_list.append(X)
    true_force_list.append(interp_cache[row][2])

X_arr = np.stack(X_list)
true_force = np.stack(true_force_list)
print(f"Built {len(X_arr)} corrected feature vectors, dim={X_arr.shape[1]}")

# ── Run the model (R10 — LiverDualAttnFlex) ──────────────────────────────────
ckpt = torch.load(f'{SOFA_ROOT}/pinn_project/train/liver_E1500_best.pth', map_location='cpu', weights_only=False)
def to_np(v): return v.cpu().numpy() if hasattr(v, 'cpu') else np.array(v)
X_mean = to_np(ckpt['X_mean'])
X_std  = to_np(ckpt['X_std'])
Y_mean = to_np(ckpt['Y_mean'])
Y_std  = to_np(ckpt['Y_std'])
model = LiverDualAttnFlex(ckpt['n_output'], ckpt['n_steps'], ckpt['n_neighbours'], ckpt['n_nb_feat'])
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"R10 model loaded: n_inputs={ckpt['n_inputs']}  feature built: {X_arr.shape[1]}")

X_norm = (X_arr - X_mean) / (X_std + 1e-8)
with torch.no_grad():
    Y_norm = model(torch.tensor(X_norm, dtype=torch.float32)).numpy()
Y_pred = Y_norm * Y_std + Y_mean
pred_force = np.sign(Y_pred[:, :3]) * (np.exp(np.abs(Y_pred[:, :3])) - 1.0)

rel_l2 = np.linalg.norm(true_force - pred_force) / (np.linalg.norm(true_force) + 1e-8)
print(f"\n=== Corrected-input prediction vs interpolated true force ===")
print(f"Relative L2 error: {rel_l2*100:.2f}%")
print(f"true |F| mean: {np.linalg.norm(true_force,axis=1).mean():.3f}  pred |F| mean: {np.linalg.norm(pred_force,axis=1).mean():.3f}")

# ── Check how much of the error is the frozen-tail period vs genuine mid-session gap ──
n_total = len(true_force)
for frac in [1.0, 0.9, 0.8, 0.7]:
    cutoff = int(n_total * frac)
    e = np.linalg.norm(true_force[:cutoff] - pred_force[:cutoff]) / (np.linalg.norm(true_force[:cutoff]) + 1e-8)
    print(f"  using first {frac*100:.0f}% of session (excluding last {100-frac*100:.0f}%): {e*100:.2f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

true_mag = np.linalg.norm(true_force, axis=1)
pred_mag = np.linalg.norm(pred_force, axis=1)
steps = np.arange(N_LAGS, N_LAGS + len(true_mag))  # real call index, showing the skip

fig, ax = plt.subplots(figsize=(11, 4))
ax.axvspan(0, N_LAGS, color='gray', alpha=0.2, label=f'skipped (first {N_LAGS} calls — no full lag window yet)')
ax.plot(steps, true_mag, 'b-o', markersize=3, label='True |F| (interpolated from real recording)')
ax.plot(steps, pred_mag, 'r-o', markersize=3, label='Corrected-input prediction |F|')
ax.set_xlabel('Replay call index')
ax.set_ylabel('|F| (N)')
sid_str = ','.join(map(str, args.session_id))
ax.set_title(f'Sessions {sid_str}: corrected-input prediction vs true force  (Rel L2: {rel_l2*100:.1f}%)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
sid_str = '_'.join(map(str, args.session_id))
outpath = f'{SOFA_ROOT}/pinn_project/test/corrected_replay_session{sid_str}.png'
plt.savefig(outpath, dpi=150)
print(f"Plot saved: {outpath}")
