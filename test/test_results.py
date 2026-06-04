#!/usr/bin/env python3
"""
Test trained PINN and visualize results.
Run: python3 test_results.py
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('/home/yogyaahuja/sofa/pinn_project/2_train')
from pinn_model import LiverPINN


# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = "/home/yogyaahuja/sofa/pinn_project/data/tissue_pinn.pth"
DATA_PATH  = "/home/yogyaahuja/sofa/pinn_project/data/training_data.csv"

print("Loading model and configuration...")
checkpoint = torch.load(MODEL_PATH, map_location='cpu')

# Use keys matching how you save your model configuration
n_output   = checkpoint['n_output']
n_vertices = checkpoint['n_vertices']
n_inputs   = checkpoint['n_inputs']
window     = checkpoint['window']
X_mean     = checkpoint['X_mean'].numpy() if isinstance(checkpoint['X_mean'], torch.Tensor) else checkpoint['X_mean']
X_std      = checkpoint['X_std'].numpy()  if isinstance(checkpoint['X_std'], torch.Tensor) else checkpoint['X_std']
vel_q_low  = checkpoint['vel_q_low'].numpy() if isinstance(checkpoint['vel_q_low'], torch.Tensor) else checkpoint['vel_q_low']
vel_q_high = checkpoint['vel_q_high'].numpy() if isinstance(checkpoint['vel_q_high'], torch.Tensor) else checkpoint['vel_q_high']
Y_scale    = checkpoint['Y_scale']        # Unpack global scaling factor
gravity_baseline = checkpoint['gravity_baseline'].numpy() if isinstance(checkpoint['gravity_baseline'], torch.Tensor) else checkpoint['gravity_baseline']

model = LiverPINN(
    n_output=n_output,
    n_inputs=n_inputs
)
model.load_state_dict(checkpoint['model_state'])
model.eval()

print(f"Model loaded. Vertices: {n_vertices}, Output: {n_output}")
print(f"Trained for {checkpoint['epoch']} epochs")
print(f"Best val loss: {checkpoint['val_loss']:.6f}")


# ============================================================
# LOAD TEST DATA
# ============================================================
df = pd.read_csv(DATA_PATH)

force_mag = np.sqrt(
    df['tool_fx'].values ** 2 +
    df['tool_fy'].values ** 2 +
    df['tool_fz'].values ** 2
)
df = df[force_mag > 0.01].reset_index(drop=True)

tool_cols = [
    'tool_x',  'tool_y',  'tool_z',
    'tool_vx', 'tool_vy', 'tool_vz',
    'tool_fx', 'tool_fy', 'tool_fz',
]
disp_cols = [c for c in df.columns
             if c.startswith('dx') or
                c.startswith('dy') or
                c.startswith('dz')]

X_raw = df[tool_cols].values.astype(np.float32)
Y_raw = df[disp_cols].values.astype(np.float32)
Y_raw = Y_raw - gravity_baseline

# Build temporal windows matching the training shift exactly
X_seq, Y_seq = [], []
for i in range(window, len(X_raw)):
    # Slice matching: i-WINDOW to i
    seq = X_raw[i-window:i].flatten()
    X_seq.append(seq)
    Y_seq.append(Y_raw[i])

X_all_windows = np.array(X_seq, dtype=np.float32)
Y_all_windows = np.array(Y_seq, dtype=np.float32)

# Slice the exact chronological final 20% 
n_test = int(len(X_all_windows) * 0.2)

X_test = X_all_windows[-n_test:]
Y_test = Y_all_windows[-n_test:]

# Normalize
vel_indices = [j for i in range(window) for j in [i*9+3, i*9+4, i*9+5]]
X_test[:, vel_indices] = np.clip(X_test[:, vel_indices], vel_q_low, vel_q_high)
X_test_norm = (X_test - X_mean) / X_std
X_tensor    = torch.FloatTensor(X_test_norm)


# ============================================================
# TEST 1 — SPEED
# ============================================================
print("\n" + "=" * 50)
print("TEST 1: INFERENCE SPEED")
print("=" * 50)

# Warm up
with torch.no_grad():
    for _ in range(10):
        _ = model(X_tensor[:1])

# Time single inference (simulates 1000Hz haptic loop)
times = []
for _ in range(1000):
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(X_tensor[:1])
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)

avg_ms = np.mean(times)
max_ms = np.max(times)
p99_ms = np.percentile(times, 99)

print(f"Average inference time: {avg_ms:.3f} ms")
print(f"99th percentile:        {p99_ms:.3f} ms")
print(f"Maximum:                {max_ms:.3f} ms")
print(f"1000 Hz budget:         1.000 ms")

if avg_ms < 1.0:
    print("RESULT: ✓ FAST ENOUGH for 1000 Hz haptics!")
else:
    print("RESULT: ✗ Too slow — needs GPU or model reduction")


# ============================================================
# TEST 2 — ACCURACY VS FEM
# ============================================================
print("\n" + "=" * 50)
print("TEST 2: ACCURACY VS FEM GROUND TRUTH")
print("=" * 50)

with torch.no_grad():
    # Re-scale model predictions from [-1, 1] back to physical meters
    Y_pred_norm = model(X_tensor).numpy()
    Y_pred      = Y_pred_norm * Y_scale

# Mean Absolute Error per vertex
mae_per_sample = np.mean(np.abs(Y_pred - Y_test), axis=1)
mae_overall    = np.mean(mae_per_sample)
max_error      = np.max(mae_per_sample)

# Root Mean Square Error
rmse = np.sqrt(np.mean((Y_pred - Y_test) ** 2))

# Relative error — normalize by displacement magnitude
disp_magnitude = np.sqrt(np.mean(Y_test ** 2)) + 1e-10
rel_error      = (rmse / disp_magnitude) * 100

print(f"Mean Absolute Error:  {mae_overall:.6f} m")
print(f"RMSE:                 {rmse:.6f} m")
print(f"Relative Error:       {rel_error:.2f}%")
print(f"Max sample error:     {max_error:.6f} m")

if rel_error < 5.0:
    print("RESULT: ✓ GOOD accuracy (< 5% relative error)")
elif rel_error < 15.0:
    print("RESULT: ~ ACCEPTABLE accuracy (5-15%)")
else:
    print("RESULT: ✗ Poor accuracy — need more data or training")


# ============================================================
# TEST 3 — PHYSICS CHECK
# ============================================================
# print("\n" + "=" * 50)
# print("TEST 3: PHYSICS SANITY CHECKS")
# print("=" * 50)

# # Fix Test 3 by providing a proper 9-column shape: [x, y, z, vx, vy, vz, fx, fy, fz]
# # Position at center with a 100N contact force along Y
# close_state = np.array([[0.0, 4.3, 0.9, 0.0, 0.0, 0.0, 0.0, -100.0, 0.0]], dtype=np.float32)

# # Position far away with zero forces acting on it
# far_state   = np.array([[15.0, 15.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

# close_norm = (close_state - X_mean) / X_std
# far_norm   = (far_state   - X_mean) / X_std

# with torch.no_grad():
#     # Scale up both checks to true spatial dimensions
#     close_deform = model(torch.FloatTensor(close_norm)).numpy() * Y_scale
#     far_deform   = model(torch.FloatTensor(far_norm)).numpy() * Y_scale

# close_mag = np.sqrt(np.mean(close_deform ** 2))
# far_mag   = np.sqrt(np.mean(far_deform   ** 2))

# print(f"Deformation at liver center: {close_mag:.6f}")
# print(f"Deformation far from liver:  {far_mag:.6f}")

# if close_mag > far_mag:
#     print("RESULT: ✓ Physically correct (more deformation near contact)")
# else:
#     print("RESULT: ✗ Physically wrong (deformation doesn't decrease with distance)")

print("\n" + "=" * 50)
print("TEST 3: PHYSICS SANITY CHECKS")
print("=" * 50)
print("Skipping manual physics sanity check for temporal model.")

# ============================================================
# VISUALIZE RESULTS
# ============================================================
print("\nGenerating result plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('PINN Training Results', fontsize=14)

# Plot 1 — Error distribution
ax = axes[0, 0]
ax.hist(mae_per_sample, bins=30, color='steelblue', edgecolor='white')
ax.set_xlabel('MAE per sample (m)')
ax.set_ylabel('Count')
ax.set_title('Error Distribution')
ax.axvline(mae_overall, color='red', linestyle='--',
           label=f'Mean: {mae_overall:.4f}')
ax.legend()

# Plot 2 — Predicted vs True for first vertex x-displacement
ax = axes[0, 1]
ax.scatter(Y_test[:, 0], Y_pred[:, 0],
           alpha=0.3, s=5, color='steelblue')
lim = max(abs(Y_test[:, 0]).max(), abs(Y_pred[:, 0]).max())
ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1)
ax.set_xlabel('FEM Ground Truth (m)')
ax.set_ylabel('PINN Prediction (m)')
ax.set_title('Prediction vs Truth\n(Vertex 0, x-direction)')

# Plot 3 — Speed histogram
ax = axes[0, 2]
ax.hist(times, bins=30, color='green', edgecolor='white')
ax.axvline(1.0, color='red', linestyle='--',
           label='1ms budget')
ax.axvline(avg_ms, color='orange', linestyle='--',
           label=f'Avg: {avg_ms:.2f}ms')
ax.set_xlabel('Inference time (ms)')
ax.set_ylabel('Count')
ax.set_title('Speed Distribution')
ax.legend()

# Plot 4 — Displacement magnitude predicted vs true
pred_mag = np.sqrt(np.mean(Y_pred.reshape(-1, n_vertices, 3)**2, axis=2))
true_mag = np.sqrt(np.mean(Y_test.reshape(-1, n_vertices, 3)**2, axis=2))

ax = axes[1, 0]
ax.plot(true_mag[0], label='FEM True', color='blue')
ax.plot(pred_mag[0], label='PINN Pred', color='orange', linestyle='--')
ax.set_xlabel('Vertex Index')
ax.set_ylabel('Displacement Magnitude (m)')
ax.set_title('Displacement Profile\n(First test sample)')
ax.legend()

# Plot 5 — Error per vertex
error_per_vertex = np.mean(
    np.abs(Y_pred - Y_test).reshape(-1, n_vertices, 3),
    axis=(0, 2))

ax = axes[1, 1]
ax.bar(range(n_vertices), error_per_vertex,
       color='steelblue', alpha=0.7)
ax.set_xlabel('Vertex Index')
ax.set_ylabel('Mean Absolute Error (m)')
ax.set_title('Error Per Vertex\n(Which vertices are hardest to predict)')

# Plot 6 — Summary table
ax = axes[1, 2]
ax.axis('off')
table_data = [
    ['Metric', 'Value', 'Target'],
    ['Avg Inference', f'{avg_ms:.2f} ms', '< 1 ms'],
    ['99th pct', f'{p99_ms:.2f} ms', '< 1 ms'],
    ['MAE', f'{mae_overall*1000:.3f} mm', '< 1 mm'],
    ['Relative Error', f'{rel_error:.1f}%', '< 10%'],
    ['Vertices', str(n_vertices), '-'],
    ['Test samples', str(n_test), '-'],
]
table = ax.table(cellText=table_data[1:],
                 colLabels=table_data[0],
                 cellLoc='center',
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)
ax.set_title('Summary', pad=20)

plt.tight_layout()
out_path = "/home/yogyaahuja/sofa/pinn_project/data/test_results.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Results saved: {out_path}")

# Print final summary
print("\n" + "=" * 50)
print("FINAL SUMMARY")
print("=" * 50)
print(f"Speed:    {avg_ms:.2f} ms avg inference")
print(f"Accuracy: {rel_error:.1f}% relative error vs FEM")
print(f"Vertices: {n_vertices}")
print(f"FEM time: ~20-50 ms per step")
print(f"Speedup:  ~{25/avg_ms:.0f}x faster than FEM")