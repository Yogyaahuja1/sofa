"""
Compare PINN predicted vs real LCP force from a live deployment run.

Reads live_pinn_vs_real.csv logged by liver_live_pinn_test.scn
(liveComparisonLog field in LCPForceFeedback).

Run after closing the live scene:
  python3 pinn_project/test/compare_live.py
"""
import os as _os
SOFA_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = f'{SOFA_ROOT}/pinn_project/data/live_pinn_vs_real.csv'
OUT_PNG  = f'{SOFA_ROOT}/pinn_project/test/live_comparison.png'

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH}")

real_fx = df['real_fx'].values
real_fy = df['real_fy'].values
real_fz = df['real_fz'].values
pinn_fx = df['pinn_fx'].values
pinn_fy = df['pinn_fy'].values
pinn_fz = df['pinn_fz'].values

f_real = np.sqrt(real_fx**2 + real_fy**2 + real_fz**2)
f_pinn = np.sqrt(pinn_fx**2 + pinn_fy**2 + pinn_fz**2)
steps  = np.arange(len(df))

# Contact mask: real force > 0.5N
contact_mask = f_real > 0.5
n_contact = contact_mask.sum()

def rel_l2(gt, pred):
    return np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-8) * 100

def mean_rel_contact(gt, pred, mask):
    if mask.sum() == 0:
        return float('nan')
    per_err = np.abs(pred[mask] - gt[mask]) / (np.abs(gt[mask]) + 1e-8)
    return per_err.mean() * 100

print(f"\n{'':>4}  {'Rel L2 (all)':>14}  {'Rel L2 (contact)':>18}  {'MAE':>10}  {'MeanRel%':>10}")
print("-" * 65)
pairs = [('Fx', real_fx, pinn_fx),
         ('Fy', real_fy, pinn_fy),
         ('Fz', real_fz, pinn_fz),
         ('|F|', f_real,  f_pinn)]

for name, gt, pred in pairs:
    a  = rel_l2(gt, pred)
    c  = rel_l2(gt[contact_mask], pred[contact_mask]) if n_contact > 0 else float('nan')
    mr = mean_rel_contact(gt, pred, contact_mask)
    mae = np.mean(np.abs(pred - gt))
    print(f"{name:>4}  {a:>13.2f}%  {c:>17.2f}%  {mae:>10.4f} N  {mr:>9.2f}%")

print(f"\nContact steps (>0.5N): {n_contact}/{len(df)} ({100*n_contact/len(df):.1f}%)")
rmse = np.sqrt(np.mean((f_pinn - f_real)**2))
print(f"|F| RMSE: {rmse:.4f} N")

# Plot
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
labels = [('Fx (N)', real_fx, pinn_fx),
          ('Fy (N)', real_fy, pinn_fy),
          ('Fz (N)', real_fz, pinn_fz),
          ('|F| (N)', f_real,  f_pinn)]

for ax, (label, gt, pred) in zip(axes, labels):
    ax.plot(steps, gt,   color='steelblue', lw=0.8, alpha=0.9, label='Real LCP (ground truth)')
    ax.plot(steps, pred, color='tomato',    lw=0.8, alpha=0.9, label='PINN (predicted)')
    ax.set_ylabel(label, fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Step', fontsize=9)
rel_contact = rel_l2(f_real[contact_mask], f_pinn[contact_mask]) if n_contact > 0 else float('nan')
fig.suptitle(f'Live deployment: PINN vs Real LCP force\n'
             f'|F| Rel L2 (contact): {rel_contact:.1f}%  |  RMSE: {rmse:.4f} N  |  Contact: {n_contact}/{len(df)} steps',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"\nPlot saved: {OUT_PNG}")
plt.show()
