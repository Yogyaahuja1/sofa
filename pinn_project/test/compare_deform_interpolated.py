"""
Same drift-correction idea as compare_force_interpolated.py, but for the
neighborhood DEFORMATION features instead of force — checking whether the
replay's freshly re-simulated FEM deformation diverges from what the original
recording shows at the same actual position, independent of position drift.

Usage:
  python3 compare_deform_interpolated.py --session-id 85
"""
import os as _os
SOFA_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--session-id', type=int, required=True)
args = parser.parse_args()

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
dx_all = sess[[f'dx{i}' for i in range(181)]].values
dy_all = sess[[f'dy{i}' for i in range(181)]].values
dz_all = sess[[f'dz{i}' for i in range(181)]].values
pdx_all = sess[[f'pdx{i}' for i in range(181)]].values
pdy_all = sess[[f'pdy{i}' for i in range(181)]].values
pdz_all = sess[[f'pdz{i}' for i in range(181)]].values
ddx_all = dx_all - pdx_all
ddy_all = dy_all - pdy_all
ddz_all = dz_all - pdz_all

pos_seq = pd.read_csv('/tmp/cpp_position_seq.csv', header=None, names=['call', 'tx', 'ty', 'tz'])
actual_pos = pos_seq[['tx', 'ty', 'tz']].values

# Parse the deform dump: call, then 20x[vertex_id, ddx, ddy, ddz]
deform_rows = []
with open('/tmp/cpp_deform_seq.csv') as f:
    for line in f:
        vals = line.strip().split(',')
        call = int(vals[0])
        rest = vals[1:]
        verts, ddxs, ddys, ddzs = [], [], [], []
        for i in range(0, len(rest), 4):
            verts.append(int(rest[i]))
            ddxs.append(float(rest[i+1]))
            ddys.append(float(rest[i+2]))
            ddzs.append(float(rest[i+3]))
        deform_rows.append((call, verts, ddxs, ddys, ddzs))

n = min(len(actual_pos), len(deform_rows))


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


actual_def_all, interp_def_all = [], []
for k in range(n):
    call, verts, ddxs, ddys, ddzs = deform_rows[k]
    i, t, d = find_bracket(actual_pos[k], ref_pos)
    for j, v in enumerate(verts):
        rec_ddx = ddx_all[i][v] + t * (ddx_all[i+1][v] - ddx_all[i][v])
        rec_ddy = ddy_all[i][v] + t * (ddy_all[i+1][v] - ddy_all[i][v])
        rec_ddz = ddz_all[i][v] + t * (ddz_all[i+1][v] - ddz_all[i][v])
        actual_def_all.append([ddxs[j], ddys[j], ddzs[j]])
        interp_def_all.append([rec_ddx, rec_ddy, rec_ddz])

actual_def_all = np.array(actual_def_all)
interp_def_all = np.array(interp_def_all)

err = np.linalg.norm(actual_def_all - interp_def_all) / (np.linalg.norm(interp_def_all) + 1e-8)
print(f"Compared {n} steps x 20 neighbor vertices = {len(actual_def_all)} deformation vectors")
print(f"Replay's actual deformation |mean|: {np.linalg.norm(actual_def_all, axis=1).mean():.5f}")
print(f"Recorded (interpolated) deformation |mean|: {np.linalg.norm(interp_def_all, axis=1).mean():.5f}")
print(f"\nRelative L2 error (replay deformation vs recorded deformation at actual position): {err*100:.2f}%")
