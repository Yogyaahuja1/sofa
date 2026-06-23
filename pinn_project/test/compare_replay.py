"""
Compare FEM ground truth vs PINN predicted forces from replay run.

FEM  source : training_data.csv   (tool_fx, tool_fy, tool_fz  — real FEM forces)
PINN source : replay_pinn_forces.csv  (fx, fy, fz — PINN predicted on same path)

Run after SOFA replay completes:
  python3 compare_replay.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FEM_CSV  = '/home/yogyaahuja/sofa/pinn_project/data/training_data.csv'
PINN_CSV = '/home/yogyaahuja/sofa/pinn_project/data/replay_pinn_forces.csv'
OUT_PNG  = '/home/yogyaahuja/sofa/pinn_project/data/replay_comparison.png'

# ── Load FEM ground truth ─────────────────────────────────────────────────────
fem = pd.read_csv(FEM_CSV, usecols=['tool_fx', 'tool_fy', 'tool_fz'])
fx_fem = fem['tool_fx'].values
fy_fem = fem['tool_fy'].values
fz_fem = fem['tool_fz'].values
f_fem  = np.sqrt(fx_fem**2 + fy_fem**2 + fz_fem**2)

# ── Load PINN replay output ───────────────────────────────────────────────────
pinn = pd.read_csv(PINN_CSV)
fx_pinn = pinn['fx'].values
fy_pinn = pinn['fy'].values
fz_pinn = pinn['fz'].values
f_pinn  = np.sqrt(fx_pinn**2 + fy_pinn**2 + fz_pinn**2)

# Match lengths (replay may finish early or late)
n = min(len(fx_fem), len(fx_pinn))
fx_fem, fy_fem, fz_fem, f_fem   = fx_fem[:n], fy_fem[:n], fz_fem[:n], f_fem[:n]
fx_pinn, fy_pinn, fz_pinn, f_pinn = fx_pinn[:n], fy_pinn[:n], fz_pinn[:n], f_pinn[:n]
steps = np.arange(n)

# ── Metrics ───────────────────────────────────────────────────────────────────
def rel_l2(gt, pred):
    return np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-8) * 100

# Only on contact steps (|FEM force| > threshold)
contact_mask = f_fem > 0.01
n_contact = contact_mask.sum()

metrics = {}
for name, gt, pred in [('Fx', fx_fem, fx_pinn),
                        ('Fy', fy_fem, fy_pinn),
                        ('Fz', fz_fem, fz_pinn),
                        ('|F|', f_fem,  f_pinn)]:
    all_l2  = rel_l2(gt, pred)
    cont_l2 = rel_l2(gt[contact_mask], pred[contact_mask]) if n_contact > 0 else float('nan')
    mae     = np.mean(np.abs(pred - gt))
    metrics[name] = (all_l2, cont_l2, mae)

print(f"\n{'':>4}  {'Rel L2 (all)':>14}  {'Rel L2 (contact)':>18}  {'MAE':>8}")
print("-" * 52)
for k, (a, c, m) in metrics.items():
    print(f"{k:>4}  {a:>13.2f}%  {c:>17.2f}%  {m:>8.4f} N")
print(f"\nContact steps: {n_contact}/{n} ({100*n_contact/n:.1f}%)")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
pairs = [('Fx (N)', fx_fem, fx_pinn),
         ('Fy (N)', fy_fem, fy_pinn),
         ('Fz (N)', fz_fem, fz_pinn),
         ('|F| (N)', f_fem,  f_pinn)]

for ax, (label, gt, pred) in zip(axes, pairs):
    ax.plot(steps, gt,   color='steelblue', lw=0.8, alpha=0.9, label='FEM (ground truth)')
    ax.plot(steps, pred, color='tomato',    lw=0.8, alpha=0.9, label='PINN (predicted)')
    ax.set_ylabel(label, fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Step', fontsize=9)
rel_contact = metrics['|F|'][1]
fig.suptitle(f'FEM vs PINN — Replay on recorded path\n'
             f'|F| Rel L2 (contact): {rel_contact:.1f}%  |  Contact steps: {n_contact}/{n}',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"\nPlot saved: {OUT_PNG}")
plt.show()
