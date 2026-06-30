"""
PINN ON vs OFF comparison for the actual C++ deployment test (replay_groundtruth.csv
vs replay_pinn.csv), plotted against cumulative distance travelled — not time —
with a full-trajectory view and a zoomed-in view of the highest-activity window.

Usage:
  python3 deployment_distance_plot.py
  python3 deployment_distance_plot.py --zoom-frac 0.3   # zoom window as fraction of total distance
"""
import os as _os
SOFA_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GT_CSV   = f'{SOFA_ROOT}/pinn_project/data/replay_groundtruth.csv'
PINN_CSV = f'{SOFA_ROOT}/pinn_project/data/replay_pinn.csv'
OUT_FULL = f'{SOFA_ROOT}/pinn_project/test/deployment_distance_full.png'
OUT_ZOOM = f'{SOFA_ROOT}/pinn_project/test/deployment_distance_zoom.png'

parser = argparse.ArgumentParser()
parser.add_argument('--zoom-frac', type=float, default=0.35,
                    help='zoom window width as a fraction of total cumulative distance')
args = parser.parse_args()

gt = pd.read_csv(GT_CSV)
pn = pd.read_csv(PINN_CSV)
n = min(len(gt), len(pn))
gt = gt.iloc[:n].reset_index(drop=True)
pn = pn.iloc[:n].reset_index(drop=True)

pos = gt[['tool_x', 'tool_y', 'tool_z']].values
step_dist = np.zeros(n)
step_dist[1:] = np.linalg.norm(np.diff(pos, axis=0), axis=1)
cum_dist = np.cumsum(step_dist)

f_gt = np.sqrt(gt.fx**2 + gt.fy**2 + gt.fz**2).values
f_pn = np.sqrt(pn.fx**2 + pn.fy**2 + pn.fz**2).values
contact = f_gt > 0.01
rel_l2 = 100 * np.linalg.norm(f_pn[contact] - f_gt[contact]) / (np.linalg.norm(f_gt[contact]) + 1e-8)

print(f"Total cumulative distance: {cum_dist[-1]:.2f} units over {n} logged steps")
print(f"Contact-region Rel L2: {rel_l2:.2f}%")

# ── FULL TRAJECTORY PLOT ──────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
fig.suptitle(f'Deployment test (C++): PINN vs Ground Truth FEM, full trajectory  |  '
             f'Rel L2 (contact): {rel_l2:.1f}%', fontsize=13, fontweight='bold')

labels = ['Fx (N)', 'Fy (N)', 'Fz (N)', '|F| (N)']
gt_cols = [gt.fx.values, gt.fy.values, gt.fz.values, f_gt]
pn_cols = [pn.fx.values, pn.fy.values, pn.fz.values, f_pn]

for i, ax in enumerate(axes):
    ax.plot(cum_dist, gt_cols[i], color='royalblue', linewidth=1.3, label='FEM (ground truth, PINN off)', alpha=0.85)
    ax.plot(cum_dist, pn_cols[i], color='orangered', linewidth=1.1, label='PINN (usePINN=true)', linestyle='--', alpha=0.85)
    ax.set_ylabel(labels[i], fontsize=10)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(loc='upper right', fontsize=9)
    if i == 3:
        ax.set_xlabel('Cumulative distance travelled (scene units)', fontsize=10)
        ax.fill_between(cum_dist, gt_cols[i], pn_cols[i], alpha=0.15, color='red')

plt.tight_layout()
plt.savefig(OUT_FULL, dpi=150, bbox_inches='tight')
print(f"Full-trajectory plot saved -> {OUT_FULL}")

# ── ZOOMED PLOT: highest-activity window, by contact force ──────────────────
window_dist = args.zoom_frac * cum_dist[-1]
# slide a window of this width across cum_dist, find the one with highest mean |F|
best_start_idx, best_mean = 0, -1
for start_idx in range(0, n, max(1, n // 200)):
    end_dist = cum_dist[start_idx] + window_dist
    end_idx = np.searchsorted(cum_dist, end_dist)
    if end_idx <= start_idx + 5:
        continue
    m = f_gt[start_idx:end_idx].mean()
    if m > best_mean:
        best_mean, best_start_idx = m, start_idx

end_idx = np.searchsorted(cum_dist, cum_dist[best_start_idx] + window_dist)
end_idx = min(end_idx, n)
sl = slice(best_start_idx, end_idx)
print(f"Zoom window: rows {best_start_idx}-{end_idx} "
      f"(distance {cum_dist[best_start_idx]:.2f} to {cum_dist[end_idx-1]:.2f})")

fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
zoom_rel_l2 = 100 * np.linalg.norm(f_pn[sl][contact[sl]] - f_gt[sl][contact[sl]]) / (np.linalg.norm(f_gt[sl][contact[sl]]) + 1e-8) if contact[sl].any() else float('nan')
fig.suptitle(f'Deployment test (C++): PINN vs Ground Truth, zoomed to highest-activity window  |  '
             f'Rel L2 here: {zoom_rel_l2:.1f}%', fontsize=13, fontweight='bold')

for i, ax in enumerate(axes):
    ax.plot(cum_dist[sl], gt_cols[i][sl], color='royalblue', linewidth=1.5,
            marker='o', markersize=3, label='FEM (ground truth, PINN off)', alpha=0.85)
    ax.plot(cum_dist[sl], pn_cols[i][sl], color='orangered', linewidth=1.3,
            marker='s', markersize=3, label='PINN (usePINN=true)', linestyle='--', alpha=0.85)
    ax.set_ylabel(labels[i], fontsize=10)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(loc='upper right', fontsize=9)
    if i == 3:
        ax.set_xlabel('Cumulative distance travelled (scene units)', fontsize=10)
        ax.fill_between(cum_dist[sl], gt_cols[i][sl], pn_cols[i][sl], alpha=0.15, color='red')

plt.tight_layout()
plt.savefig(OUT_ZOOM, dpi=150, bbox_inches='tight')
print(f"Zoomed plot saved -> {OUT_ZOOM}")
