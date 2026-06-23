"""
Export trained PINN model for C++ integration.
Produces:
  pinn_model_traced.pt       — TorchScript model (loaded by PINNPredictor)
  normalization_stats.csv    — X/Y mean+std (read by PINNPredictor)
  liver_vertices.csv         — rest-pose vertex positions (read by PINNPredictor)

Run once after training:
  python3 export_for_cpp.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'train'))

import torch
import numpy as np
from pinn_model import LiverUNet

BEST_PATH    = '/home/yogyaahuja/sofa/pinn_project/train/tissue_pinn_force_final_best.pth'
FULL_PATH    = '/home/yogyaahuja/sofa/pinn_project/train/tissue_pinn_force_final.pth'
VERTICES_NPY = '/home/yogyaahuja/sofa/pinn_project/data/liver_vertices.npy'
OUT_DIR      = '/home/yogyaahuja/sofa/pinn_project/cpp'

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load checkpoint ────────────────────────────────────────────────────────────
# Best checkpoint is state_dict only (saved during training loop).
# Load normalization stats from the full final checkpoint.
print("Loading checkpoint...")
full_ckpt = torch.load(FULL_PATH, map_location='cpu', weights_only=False)
n_in  = full_ckpt['n_inputs']
n_out = full_ckpt['n_output']

def to_np(v):
    return v.cpu().numpy() if hasattr(v, 'numpy') else np.array(v)

X_mean = to_np(full_ckpt['X_mean'])
X_std  = to_np(full_ckpt['X_std'])
Y_mean = to_np(full_ckpt['Y_mean'])
Y_std  = to_np(full_ckpt['Y_std'])
print(f"  n_inputs={n_in}, n_outputs={n_out}")
print(f"  X_mean shape: {X_mean.shape}, Y_mean shape: {Y_mean.shape}")

# Load best weights (state_dict only)
best_state = torch.load(BEST_PATH, map_location='cpu', weights_only=False)
# If it's accidentally a full dict, unwrap it
if isinstance(best_state, dict) and 'model_state' in best_state:
    best_state = best_state['model_state']

model = LiverUNet(n_output=n_out, n_inputs=n_in)
model.load_state_dict(best_state)
model.eval()

# ── Export TorchScript ─────────────────────────────────────────────────────────
print("Tracing model to TorchScript...")
example = torch.zeros(1, n_in)
with torch.no_grad():
    traced = torch.jit.trace(model, example)

out_pt = os.path.join(OUT_DIR, 'pinn_model_traced.pt')
traced.save(out_pt)
print(f"  Saved: {out_pt}")

# Quick sanity check
loaded = torch.jit.load(out_pt)
loaded.eval()
with torch.no_grad():
    out1 = model(example)
    out2 = loaded(example)
    diff = (out1 - out2).abs().max().item()
print(f"  Trace sanity check max diff: {diff:.2e}  (should be ~0)")

# ── Export normalization stats ─────────────────────────────────────────────────
out_norm = os.path.join(OUT_DIR, 'normalization_stats.csv')
with open(out_norm, 'w') as f:
    f.write('x_mean,' + ','.join(f'{v:.10f}' for v in X_mean) + '\n')
    f.write('x_std,'  + ','.join(f'{v:.10f}' for v in X_std)  + '\n')
    f.write('y_mean,' + ','.join(f'{v:.10f}' for v in Y_mean) + '\n')
    f.write('y_std,'  + ','.join(f'{v:.10f}' for v in Y_std)  + '\n')
print(f"  Saved: {out_norm}")

# ── Export vertex positions ────────────────────────────────────────────────────
print("Exporting vertex positions...")
verts = np.load(VERTICES_NPY)   # shape (181, 3)
out_verts = os.path.join(OUT_DIR, 'liver_vertices.csv')
with open(out_verts, 'w') as f:
    f.write(f"{len(verts)}\n")
    for v in verts:
        f.write(f"{v[0]:.10f},{v[1]:.10f},{v[2]:.10f}\n")
print(f"  Saved: {out_verts}  ({len(verts)} vertices)")

print("\nDone. Files ready for C++ integration:")
print(f"  {out_pt}")
print(f"  {out_norm}")
print(f"  {out_verts}")
