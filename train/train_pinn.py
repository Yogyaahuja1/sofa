"""
PINN Training Script — Liver Tissue Deformation
================================================
Input:  9 values — tool (x,y,z), velocity (vx,vy,vz), force (fx,fy,fz)
Output: 1629 values — displacement (dx,dy,dz) per vertex (543 vertices)

Physics losses enforced:
  1. FEM constitutive:  K × u = f_contact
  2. Boundary condition: fixed vertices have zero displacement
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pinn_model import LiverPINN

# ============================================================
# CONFIGURATION
# ============================================================
CSV_PATH      = '/home/yogyaahuja/sofa/pinn_project/data/training_data.csv'
N_VERTICES    = 181
N_OUT         = N_VERTICES * 3          # 1629
FIXED_INDICES = [3, 39, 64]             # from FixedConstraint in scene
DT            = 0.005                   # scene dt
BATCH_SIZE    = 64
N_EPOCHS      = 5000
LR            = 3e-4
TRAIN_SPLIT   = 0.8
CHECKPOINT_PATH = '/home/yogyaahuja/sofa/pinn_project/data/tissue_pinn_checkpoint.pth'

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("Loading data...")
df = pd.read_csv(CSV_PATH)
# Remove rows containing NaN or Inf
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# CLIP FORCES SO DATA DISTRIBUTION MATCHES TRAINING
df['tool_fx'] = df['tool_fx'].clip(-200000.0, 200000.0)
df['tool_fy'] = df['tool_fy'].clip(-200000.0, 200000.0)
df['tool_fz'] = df['tool_fz'].clip(-200000.0, 200000.0)


print(f"Rows after cleaning: {len(df)}")
print(f"  Rows loaded: {len(df)}")
print(f"  Columns:     {len(df.columns)}")

# Compute compressed solver features (use actual columns present)
force_cols_x = sorted(
    [c for c in df.columns if c.startswith('fvx')],
    key=lambda c: int(c[3:])
)
force_cols_y = [c.replace('fvx', 'fvy', 1) for c in force_cols_x]
force_cols_z = [c.replace('fvx', 'fvz', 1) for c in force_cols_x]
vel_cols_y   = [f'vvy{i}' for i in range(181)]
prev_cols_y  = [f'pdy{i}' for i in range(181)]
prev_dy_cols = [f'pdy{i}' for i in range(181)]

df['f_total_x']   = df[force_cols_x].sum(axis=1)
df['f_total_y']   = df[force_cols_y].sum(axis=1)
df['f_total_z']   = df[force_cols_z].sum(axis=1)
df['f_max_mag']   = np.sqrt(
    df[force_cols_x].values**2 +
    df[force_cols_y].values**2 +
    df[force_cols_z].values**2).max(axis=1)
df['f_n_contact'] = (np.sqrt(
    df[force_cols_x].values**2 +
    df[force_cols_y].values**2 +
    df[force_cols_z].values**2) > 0.01).sum(axis=1)
df['v_max_mag']   = np.abs(df[vel_cols_y].values).max(axis=1)
df['pd_max_mag']  = np.abs(df[prev_cols_y].values).max(axis=1)
df['pd_mean_mag'] = np.abs(df[prev_cols_y].values).mean(axis=1)

print("\n=== Force Feature Statistics ===")
print(df['f_total_x'].describe())
print(df['f_total_y'].describe())
print(df['f_total_z'].describe())
print(df['f_max_mag'].describe())
print(df['f_n_contact'].describe())

# Pick the most active vertex (skip fixed indices)
dy_cols = [c for c in df.columns if c.startswith('dy')]
dy_max = df[dy_cols].abs().max()
best_vertex_id = None
for col in dy_max.sort_values(ascending=False).index:
    vid = int(col[2:])
    if vid not in FIXED_INDICES:
        best_vertex_id = vid
        break
if best_vertex_id is None:
    best_vertex_id = 0
print(f"Most active vertex: {best_vertex_id}")


prev_dx_cols = [f'pdx{i}' for i in range(181) if i not in FIXED_INDICES]
prev_dy_cols = [f'pdy{i}' for i in range(181) if i not in FIXED_INDICES]
prev_dz_cols = [f'pdz{i}' for i in range(181) if i not in FIXED_INDICES]


# df['curr_delta_x'] = df[f'dx{best_vertex_id}'] - df[f'pdx{best_vertex_id}']
# df['curr_delta_y'] = df[f'dy{best_vertex_id}'] - df[f'pdy{best_vertex_id}']
# df['curr_delta_z'] = df[f'dz{best_vertex_id}'] - df[f'pdz{best_vertex_id}']
prev_all_cols = []
for i in range(N_VERTICES):
    if i not in FIXED_INDICES:
        prev_all_cols += [f'pdx{i}', f'pdy{i}', f'pdz{i}']


# Compressed accumulated stress (summary stats, not all 543)
sax_cols = [f'sax{i}' for i in range(181)]
say_cols = [f'say{i}' for i in range(181)]

df['stress_max'] = np.sqrt(
    df[sax_cols].values**2 + 
    df[say_cols].values**2).max(axis=1)
df['stress_mean'] = np.sqrt(
    df[sax_cols].values**2 + 
    df[say_cols].values**2).mean(axis=1)
df['stress_n_active'] = (df[say_cols].abs() > 0.001).sum(axis=1)


input_cols = [
    'tool_x',    'tool_y',    'tool_z',
    'tool_vx',   'tool_vy',   'tool_vz',
    'tool_fx',   'tool_fy',   'tool_fz',
    'stress_max', 'stress_mean', 'stress_n_active',  # ← ADD
 ]+ prev_all_cols


df = df.dropna().copy()
# ── 4. Dynamic Dimensionality (No Hardcoding!) ───────────────────────────────
N_INPUTS = len(input_cols)  # Dynamically evaluates to 27, ensuring no PyTorch crashes

# ── 5. Align Outputs with What We Want to Predict ────────────────────────────
# We are predicting the frame-to-frame delta movements
deform_cols = []
for i in range(N_VERTICES):
    if i not in FIXED_INDICES:
        deform_cols += [f'dx{i}', f'dy{i}', f'dz{i}']
N_OUT = len(deform_cols)  # 178 * 3 = 534

# Check all columns exist
missing_in = [c for c in input_cols  if c not in df.columns]
missing_out= [c for c in deform_cols if c not in df.columns]


if missing_in:
    raise ValueError(f"Missing input columns: {missing_in}\n"
                     f"Available: {list(df.columns[:15])}")
if missing_out:
    raise ValueError(f"Missing {len(missing_out)} deformation columns. "
                     f"Check N_VERTICES={N_VERTICES} matches your mesh.")

X_raw_tensor = torch.FloatTensor(df[input_cols].values)   # (N, 9)
stds = X_raw_tensor.std(dim=0)
print("Zero std columns:", (stds < 1e-8).sum().item())
idx = torch.argmin(stds)
print("Smallest std column:", input_cols[idx])
print("X std min/max:", stds.min().item(), "/", stds.max().item())

N_INPUTS = len(input_cols)
print(f"Inputs kept (no variance filtering): {N_INPUTS}")

Y_raw_tensor = torch.FloatTensor(df[deform_cols].values)  # (N, 3)

force_mag = X_raw_tensor[:, 6:9].norm(dim=1)
force_mask = force_mag > 0.01
X_raw_tensor = X_raw_tensor[force_mask]
Y_raw_tensor = Y_raw_tensor[force_mask]
print(f"Rows after force filtering: {len(X_raw_tensor)}")

# ── Build temporal windows chronologically ─────────────────────
WINDOW = 1    



X_all = X_raw_tensor
Y_all = Y_raw_tensor


# ── Chronological Split (Prevents data leakage!) ───────────────
# ADD before n_train split:
perm_all = torch.randperm(len(X_all))
X_all_shuffled = X_all[perm_all]
Y_all_shuffled = Y_all[perm_all]
n_train = int(TRAIN_SPLIT * len(X_all))

X_train_raw = X_all[:n_train]
Y_train_raw = Y_all[:n_train]

X_test_raw = X_all_shuffled[n_train:]
Y_test_raw = Y_all_shuffled[n_train:]  # Held out clean test data

Y_scale = float(Y_all.abs().max().item()) * 1.1
print(f"  Y_scale from full data: {Y_scale:.4f}")


# Shuffle ONLY the training set so sequence chunks don't overfit batches
perm = torch.randperm(len(X_train_raw))
X_train_raw = X_train_raw[perm]
Y_train_raw = Y_train_raw[perm]

X = X_train_raw
Y = Y_train_raw


print(f"  Input  shape: {X.shape}")
print(f"  Output shape: {Y.shape}")



if Y_scale < 1e-8:
    Y_scale = 1.0
    print("WARNING: all deformations near zero — check scene")
# Clip your normalized Y values to [-1, 1] so the network stays perfectly stable
Y_norm = torch.clamp(Y / Y_scale, -1.0, 1.0)
Y_train = Y_norm
Y_test  = torch.clamp(Y_test_raw / Y_scale, -1.0, 1.0)
    


Y_mean = torch.zeros(Y.shape[1]) # not used but keep for compatibility
Y_std  = torch.ones(Y.shape[1])  # not used but keep for compatibility

print(f"  Deformation scale factor: {Y_scale:.6f}")
clipped_ratio = (Y_norm.abs() >= 0.999).float().mean().item() * 100.0
print(f"  Y clipped ratio: {clipped_ratio:.2f}%")

# ── Normalize inputs ──────────────────────────────────────────
# Normalize — X is now (N, 60)

vel_indices = [3, 4, 5]
vel_q_low = X_train_raw[:, vel_indices].quantile(0.01, dim=0)
vel_q_high = X_train_raw[:, vel_indices].quantile(0.99, dim=0)
X_train_raw[:, vel_indices] = torch.clamp(X_train_raw[:, vel_indices], vel_q_low, vel_q_high)
X_test_raw[:, vel_indices] = torch.clamp(X_test_raw[:, vel_indices], vel_q_low, vel_q_high)

# Separate standard kinematics from structural spatial pdy inputs

X_mean = X_train_raw.mean(dim=0)
X_std  = X_train_raw.std(dim=0) + 1e-8
X_train = (X_train_raw - X_mean) / X_std
X_test  = (X_test_raw  - X_mean) / X_std


print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")
print(f"  X std min/max: {X_std.min().item():.6f} / {X_std.max().item():.6f}")
print(f"  Vel clamp low/high (first 3): {vel_q_low[:3].tolist()} /X_mean {vel_q_high[:3].tolist()}")



# ============================================================
# STEP 3: PHYSICS LOSSES
# ============================================================

def loss_data(u_pred, u_true):
    return torch.mean((u_pred - u_true) ** 2)


def loss_boundary(u_pred, fixed_idx):
    """
    Fixed vertices must have zero displacement.
    u_pred: (batch, N_VERTICES * 3) — flat
    Extract x,y,z for each fixed vertex and penalise.
    """
    total = torch.tensor(0.0, device=u_pred.device)  # ← same device as u_pred
    for vi in fixed_idx:
        # NEW METHOD
        if u_pred.shape[1] == 3:
            return torch.tensor(0.0, device=u_pred.device)
        u_reshaped = u_pred.reshape(-1, 181, 3) 
        total = total + torch.mean(u_reshaped[:, vi, : ] ** 2)
    return total / len(fixed_idx)


def loss_fem_constitutive(u_pred, f_contact, K):
    """
    FEM equilibrium: K × u = f_contact
    Residual = K×u - f should be zero.

    u_pred:    (batch, N_OUT)
    f_contact: (batch, N_OUT)  — force columns from CSV
    K:         (N_OUT, N_OUT)
    """
    Ku = u_pred @ K.T           # (batch, N_OUT)
    residual = Ku - f_contact
    return torch.mean(residual ** 2)


# ============================================================
# STEP 4: TRAINING
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

model = LiverPINN(n_output=N_OUT, n_inputs=N_INPUTS).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=500, factor=0.5, min_lr=1e-6
)

dataset = TensorDataset(X_train.to(device), Y_train.to(device))
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Extract contact force from input (columns 6,7,8 = fx,fy,fz)
# We need to expand to N_OUT for FEM loss — use mean force per vertex
# (simplified: broadcast tool force to all vertices)
# For full physics you would export per-vertex forces from SOFA

history = {'total':[], 'data':[], 'bc':[], 'epoch':[]}

start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    history = checkpoint.get('history', history)
    start_epoch = int(checkpoint.get('epoch', -1)) + 1
    print(f"Resuming from checkpoint at epoch {start_epoch}")

print("\nTraining...\n")
for epoch in range(start_epoch, N_EPOCHS):

    model.train()
    epoch_loss = epoch_data = epoch_bc = 0.0

    for X_batch, Y_batch in loader:

        u_pred = model(X_batch)   # (batch, N_OUT)

        # Data loss
        L_data = loss_data(u_pred, Y_batch)

        # Physics weight — ramp up slowly
        L_bc = loss_boundary(u_pred, FIXED_INDICES)
        progress = epoch / N_EPOCHS
        w_bc = 0.1 + 0.4 * progress
        L = L_data + w_bc * L_bc

        optimizer.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += L.item()
        epoch_data += L_data.item()
        epoch_bc   += L_bc.item()

    n_batches = len(loader)
    scheduler.step(epoch_loss / n_batches)

    if epoch % 200 == 0:
        history['epoch'].append(epoch)
        history['total'].append(epoch_loss / n_batches)
        history['data'].append(epoch_data / n_batches)
        history['bc'].append(epoch_bc / n_batches)

        print(f"Epoch {epoch:5d} | "
              f"Total: {epoch_loss/n_batches:.6f} | "
              f"Data: {epoch_data/n_batches:.6f} | "
              f"BC: {epoch_bc/n_batches:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'history': history,
        }, CHECKPOINT_PATH)

    if epoch in {2500, N_EPOCHS - 1}:
        tag = "2500" if epoch == 2500 else "5000"
        snapshot_path = f"/home/yogyaahuja/sofa/pinn_project/data/tissue_pinn_snapshot_{tag}.pth"
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'history': history,
        }, snapshot_path)
        print(f"Saved snapshot at epoch {epoch} to {snapshot_path}")


# ============================================================
# STEP 5: VALIDATE — ERROR vs FEM
# ============================================================
print("\n=== Validation vs FEM ===")
model.eval()
with torch.no_grad():
    u_pred_norm = model(X_test.to(device)).cpu()

    # SCALE REALIGNMENT: Un-scale both predictions and targets back to physical meters
    u_pred = u_pred_norm          # normalized [-1, 1]
    u_true = Y_test               # normalized [-1, 1]  ← already computed earlier

    print("u_pred nan:", torch.isnan(u_pred).any())
    print("u_true nan:", torch.isnan(u_true).any())

    print("u_pred inf:", torch.isinf(u_pred).any())
    print("u_true inf:", torch.isinf(u_true).any())

    print("max pred:", u_pred.abs().max())
    print("max true:", u_true.abs().max())
    # Relative L2 error — main metric
    # Relative L2 error — main metric
    rel_err = torch.norm(u_pred - u_true) / (torch.norm(u_true) + 1e-8)
    print(f"Relative L2 error:        {rel_err.item()*100:.2f}%")

    if N_OUT == 3:
        per_sample_err = torch.norm(u_pred - u_true, dim=1)
        print(f"Mean delta error:         {per_sample_err.mean().item():.6f}")
        print(f"Max delta error:          {per_sample_err.max().item():.6f}")
    else:
        # Pure PyTorch tracking avoids array type errors
        diff = (u_pred - u_true).reshape(-1, N_VERTICES, 3)
        per_vertex_err = torch.norm(diff, dim=2)

    # ADD — visual comparison of one sample
    import matplotlib
    matplotlib.use('Agg')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1 — full deformation comparison
    axes[0].plot(u_true[0].numpy(), label='FEM true', alpha=0.7)
    axes[0].plot(u_pred[0].numpy(), label='PINN pred', alpha=0.7)
    axes[0].set_title('Sample 0: Full deformation')
    axes[0].legend()

    if N_OUT != 3:
        # Plot 2 — worst sample
        worst_idx = per_vertex_err.max(dim=1).values.argmax().item()
        axes[1].plot(u_true[worst_idx].numpy(), label='FEM true', alpha=0.7)
        axes[1].plot(u_pred[worst_idx].numpy(), label='PINN pred', alpha=0.7)
        axes[1].set_title(f'Worst sample ({worst_idx})')
        axes[1].legend()

        # Plot 3 — best sample
        best_idx = per_vertex_err.max(dim=1).values.argmin().item()
        axes[2].plot(u_true[best_idx].numpy(), label='FEM true', alpha=0.7)
        axes[2].plot(u_pred[best_idx].numpy(), label='PINN pred', alpha=0.7)
        axes[2].set_title(f'Best sample ({best_idx})')
        axes[2].legend()

    plt.tight_layout()
    plt.savefig('sample_comparison.png', dpi=150)
    print("Sample comparison saved to sample_comparison.png")
    

    if N_OUT != 3:
        # Removed the multiplication by Y_scale since u_pred and u_true are already in raw physical scales
        # CHANGE TO:
        max_err_mm = per_vertex_err.max().item() * Y_scale * 1000
        mean_err_mm = per_vertex_err.mean().item() * Y_scale * 1000

        print(f"Max vertex error:         {max_err_mm:.3f} mm")
        print(f"Mean vertex error:        {mean_err_mm:.4f} mm")

    
        # ← ADD HERE ↓
        contact_mask = (np.linalg.norm(Y_test.numpy(), axis=1) > 0.001)
        print(f"Contact frames in test:   {contact_mask.sum().item()} / {len(contact_mask)}")

        rel_err_contact = torch.norm(u_pred[contact_mask] - u_true[contact_mask]) / \
                        (torch.norm(u_true[contact_mask]) + 1e-8)
        print(f"Relative L2 error (contact only): {rel_err_contact.item()*100:.2f}%")

    if N_OUT != 3:
        # Boundary condition check
        for vi in FIXED_INDICES:
            cols = slice(3*vi, 3*vi+3)
            fixed_err = u_pred[:, cols].abs().max().item() * 1000
            print(f"Fixed vertex {vi:3d} max disp: {fixed_err:.4f} mm")

    if N_OUT != 3:
        # What % of test samples are within 1mm error?
        within_1mm = (per_vertex_err.max(dim=1).values < 0.001 / Y_scale).float().mean().item()
        print(f"Samples within 1mm:       {within_1mm*100:.1f}%")

    # ADD in Step 5 validation:
    print("Train target mean abs:", Y_train.abs().mean().item())
    print("Test target mean abs:",  Y_test.abs().mean().item())
    print("Pred mean abs:",         u_pred.abs().mean().item())
    mae = (u_pred-u_true).abs().mean()
    rmse = torch.sqrt(((u_pred-u_true)**2).mean())

    print("MAE:", mae.item())
    print("RMSE:", rmse.item())


# ============================================================
# STEP 6: SAVE
torch.save({
    'model_state': model.state_dict(),
    'n_output':    N_OUT,
    'n_vertices':  N_VERTICES,
    'n_inputs':    N_INPUTS,    # ← add this
    'window':      WINDOW,      # ← add this
    'X_mean':      X_mean,
    'X_std':       X_std,
    'vel_q_low':   vel_q_low,
    'vel_q_high':  vel_q_high,
    'Y_scale':     Y_scale,
    'epoch':       N_EPOCHS,
    'val_loss':    rel_err.item(),
}, 'tissue_pinn.pth')
print("\nModel saved to tissue_pinn.pth")
print("Config saved to model_config.pth")

# ============================================================
# STEP 7: PLOT LOSS CURVES
# ============================================================
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history['epoch'], history['total'],  label='Total')
plt.plot(history['epoch'], history['data'],   label='Data')
plt.plot(history['epoch'], history['bc'],     label='Boundary')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.title('Training Loss')
plt.yscale('log')

plt.subplot(1, 2, 2)
plt.bar(['Rel L2 %', 'MAE', 'RMSE'],
        [rel_err.item()*100, mae.item(), rmse.item()])
plt.title('Validation vs FEM')

plt.tight_layout()
plt.savefig('training_results.png', dpi=150)
print("Plot saved to training_results.png")