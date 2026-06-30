"""
Sir's drift-corrected comparison: instead of comparing PINN's prediction against
the recorded force at the SAME ROW INDEX (which assumes the replay's actual state
exactly matches that row's recorded position — proven false earlier today, the
replay drifts), compare against the recorded force AT THE EXACT POSITION the
replay's real physics actually reached. This separates "is the model's force
prediction correct, given wherever the tool really is" from "did the replay
simulation reproduce the original recording" — two different questions that the
naive row-index comparison conflates into one number.

How: for each step, take the replay's ACTUAL instrument position (dumped by
PINNPredictor to /tmp/cpp_position_seq.csv — not the commanded/logged position,
the real one used to compute the prediction). Project that position onto the
original recorded path (as a polyline, not just nearest-row) and linearly
interpolate the recorded force along the closest segment. Compare PINN's
predicted force (from replay_pinn.csv) against this interpolated reference.

Usage:
  python3 compare_force_interpolated.py --session-id 85
"""
import os as _os
SOFA_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--session-id', type=int, required=True)
args = parser.parse_args()

# ── Load the original recorded path for this session (the interpolation reference) ──
df = pd.read_csv(f'{SOFA_ROOT}/pinn_project/data/training_data.csv')
sess = df[df.session_id == args.session_id].sort_values('step').reset_index(drop=True)
gaps = sess['step'].diff().abs()
gap_idx = list(gaps[gaps > 50].index)
if gap_idx:
    bounds = [0] + gap_idx + [len(sess)]
    blocks = [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]
    best = max(blocks, key=lambda b: b[1]-b[0])
    sess = sess.iloc[best[0]:best[1]].reset_index(drop=True)

ref_pos = sess[['tool_x', 'tool_y', 'tool_z']].values
ref_force = sess[['tool_fx', 'tool_fy', 'tool_fz']].values
n_ref = len(ref_pos)
print(f"Reference path: {n_ref} recorded rows")

# ── Load the replay's ACTUAL position at each prediction call ──────────────────
pos_seq = pd.read_csv('/tmp/cpp_position_seq.csv', header=None, names=['call', 'tx', 'ty', 'tz'])
actual_pos = pos_seq[['tx', 'ty', 'tz']].values

# ── Load PINN's predicted force, one value per call (every 3rd logged row) ─────
replay = pd.read_csv(f'{SOFA_ROOT}/pinn_project/data/replay_pinn.csv')
pinn_force_all = replay[['fx', 'fy', 'fz']].values[::3]

n = min(len(actual_pos), len(pinn_force_all))
actual_pos = actual_pos[:n]
pinn_force = pinn_force_all[:n]


def closest_point_on_polyline(query, path):
    """Project query onto the path (piecewise-linear), return interpolated
    force at the closest point on the closest segment, and the distance."""
    best_dist = np.inf
    best_force = None
    for i in range(len(path) - 1):
        a, b = path[i], path[i+1]
        ab = b - a
        ab_len2 = np.dot(ab, ab)
        if ab_len2 < 1e-12:
            t = 0.0
        else:
            t = np.clip(np.dot(query - a, ab) / ab_len2, 0.0, 1.0)
        proj = a + t * ab
        dist = np.linalg.norm(query - proj)
        if dist < best_dist:
            best_dist = dist
            best_force = ref_force[i] + t * (ref_force[i+1] - ref_force[i])
    return best_force, best_dist


interp_force = np.zeros((n, 3))
proj_dist = np.zeros(n)
for i in range(n):
    f, d = closest_point_on_polyline(actual_pos[i], ref_pos)
    interp_force[i] = f
    proj_dist[i] = d

# ── Drift-corrected comparison: PINN vs force AT THE ACTUAL POSITION REACHED ──
err_corrected = np.linalg.norm(pinn_force - interp_force) / (np.linalg.norm(interp_force) + 1e-8)

# ── Naive comparison for contrast: PINN vs recorded force at the SAME ROW INDEX ──
naive_n = min(n, n_ref)
naive_ref = ref_force[:naive_n]
err_naive = np.linalg.norm(pinn_force[:naive_n] - naive_ref) / (np.linalg.norm(naive_ref) + 1e-8)

print(f"\nAverage projection distance (replay drift from recorded path): {proj_dist.mean():.4f}  max: {proj_dist.max():.4f}")
print(f"\nNAIVE comparison (PINN vs recorded force at same row index):      {err_naive*100:.2f}%")
print(f"DRIFT-CORRECTED comparison (PINN vs force at actual position):    {err_corrected*100:.2f}%")
print(f"\n(If corrected << naive: most of the error was replay drift, not the model.)")
print(f"(If corrected ~= naive: the model itself is genuinely off, not just drift.)")
