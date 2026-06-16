import torch
import torch.nn as nn
import numpy as np

import re
import numpy as np

with open('/home/yogyaahuja/sofa/build/bin/liver_physics.txt') as f:
    txt = f.read()

nums = [float(x) for x in re.findall(
    r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', txt
)]

K_numpy = np.array(nums, dtype=np.float32).reshape(543, 543)

print("K shape =", K_numpy.shape)

# Verify the shape is exactly (543, 543)
print(f"Loaded K matrix shape: {K_numpy.shape}")
assert K_numpy.shape == (543, 543), "Matrix shape is wrong!"

# 2. Convert to PyTorch Tensor and move to your device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_tensor = torch.tensor(K_numpy, dtype=torch.float32).to(device)

class LiverPINN(nn.Module):
    def __init__(self, n_output: int, n_inputs: int = 611):  # ← add n_inputs
        super().__init__()
        self.n_output = n_output

        # # Fourier encoding — now maps 90D input
        # self.n_fourier = 256
        # self.register_buffer('B', torch.randn(n_inputs, self.n_fourier) * 2)

        # n_in = 2 * self.n_fourier  # 128

        self.net = nn.Sequential(
            nn.Linear(n_inputs, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, n_output)
        )

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    # def fourier_encode(self, x):
    #     proj = 2 * np.pi * (x @ self.B)
    #     return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    # ADD in pinn_model.py forward():
    def forward(self, x):
        # out = self.net(self.fourier_encode(x))
        # return out
        return self.net(x)

class LiverUNet(nn.Module):
    """MLP with U-Net style encoder/decoder + skip connections."""
    def __init__(self, n_output: int, n_inputs: int = 610):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(nn.Linear(n_inputs, 512), nn.LayerNorm(512), nn.GELU())
        self.enc2 = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU())
        self.enc3 = nn.Sequential(nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU())

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU())

        # Decoder with skip connections
        self.dec3 = nn.Sequential(nn.Linear(64 + 128, 128), nn.LayerNorm(128), nn.GELU())
        self.dec2 = nn.Sequential(nn.Linear(128 + 256, 256), nn.LayerNorm(256), nn.GELU())
        self.dec1 = nn.Sequential(nn.Linear(256 + 512, 512), nn.LayerNorm(512), nn.GELU())

        self.output = nn.Linear(512, n_output)

    def forward(self, x):
        e1 = self.enc1(x)      # (batch, 512)
        e2 = self.enc2(e1)     # (batch, 256)
        e3 = self.enc3(e2)     # (batch, 128)

        b = self.bottleneck(e3)  # (batch, 64)

        d3 = self.dec3(torch.cat([b,  e3], dim=1))  # (batch, 128)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # (batch, 256)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # (batch, 512)

        return self.output(d1)  # (batch, n_output)


class ResBlock(nn.Module):
    """Two linear layers + LayerNorm/GELU with an internal residual skip."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1   = nn.Linear(in_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.fc2   = nn.Linear(out_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.act   = nn.GELU()
        self.proj  = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        h = self.act(self.norm1(self.fc1(x)))
        h = self.norm2(self.fc2(h))
        return self.act(h + self.proj(x))


class LiverResUNet(nn.Module):
    """Residual U-Net: same encoder/decoder/skip-connection layout as
    LiverUNet, but each stage is itself a residual block (two linear
    layers + internal skip) for deeper effective depth and better
    gradient flow."""
    def __init__(self, n_output: int, n_inputs: int = 610):
        super().__init__()

        # Encoder
        self.enc1 = ResBlock(n_inputs, 512)
        self.enc2 = ResBlock(512, 256)
        self.enc3 = ResBlock(256, 128)

        # Bottleneck
        self.bottleneck = ResBlock(128, 64)

        # Decoder with skip connections
        self.dec3 = ResBlock(64 + 128, 128)
        self.dec2 = ResBlock(128 + 256, 256)
        self.dec1 = ResBlock(256 + 512, 512)

        self.output = nn.Linear(512, n_output)

    def forward(self, x):
        e1 = self.enc1(x)      # (batch, 512)
        e2 = self.enc2(e1)     # (batch, 256)
        e3 = self.enc3(e2)     # (batch, 128)

        b = self.bottleneck(e3)  # (batch, 64)

        d3 = self.dec3(torch.cat([b,  e3], dim=1))  # (batch, 128)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # (batch, 256)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # (batch, 512)

        return self.output(d1)  # (batch, n_output)


class AttentionGate(nn.Module):
    """Gates skip-connection features using the decoder's incoming state —
    learns how much of each encoder feature to pass through, instead of
    blindly concatenating everything."""
    def __init__(self, skip_dim, gate_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or min(skip_dim, gate_dim)
        self.w_gate = nn.Linear(gate_dim, hidden_dim)
        self.w_skip = nn.Linear(skip_dim, hidden_dim)
        self.psi    = nn.Linear(hidden_dim, skip_dim)
        self.act    = nn.ReLU()

    def forward(self, skip, gate):
        attn = torch.sigmoid(self.psi(self.act(self.w_gate(gate) + self.w_skip(skip))))
        return skip * attn


class LiverAttentionUNet(nn.Module):
    """Same encoder/decoder layout as LiverUNet, but each skip connection
    passes through an AttentionGate (gated by the decoder's current state)
    before being concatenated into the decoder."""
    def __init__(self, n_output: int, n_inputs: int = 610):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(nn.Linear(n_inputs, 512), nn.LayerNorm(512), nn.GELU())
        self.enc2 = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU())
        self.enc3 = nn.Sequential(nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU())

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU())

        # Attention gates for each skip connection
        self.attn3 = AttentionGate(skip_dim=128, gate_dim=64)
        self.attn2 = AttentionGate(skip_dim=256, gate_dim=128)
        self.attn1 = AttentionGate(skip_dim=512, gate_dim=256)

        # Decoder with gated skip connections
        self.dec3 = nn.Sequential(nn.Linear(64 + 128, 128), nn.LayerNorm(128), nn.GELU())
        self.dec2 = nn.Sequential(nn.Linear(128 + 256, 256), nn.LayerNorm(256), nn.GELU())
        self.dec1 = nn.Sequential(nn.Linear(256 + 512, 512), nn.LayerNorm(512), nn.GELU())

        self.output = nn.Linear(512, n_output)

    def forward(self, x):
        e1 = self.enc1(x)      # (batch, 512)
        e2 = self.enc2(e1)     # (batch, 256)
        e3 = self.enc3(e2)     # (batch, 128)

        b = self.bottleneck(e3)  # (batch, 64)

        d3 = self.dec3(torch.cat([b, self.attn3(e3, b)], dim=1))   # (batch, 128)
        d2 = self.dec2(torch.cat([d3, self.attn2(e2, d3)], dim=1)) # (batch, 256)
        d1 = self.dec1(torch.cat([d2, self.attn1(e1, d2)], dim=1)) # (batch, 512)

        return self.output(d1)  # (batch, n_output)


class LiverTransformer(nn.Module):
    """Self-attention over the 20 nearest-neighbour vertex tokens + 1 global
    tool-kinematics token. Mirrors FEM-style deformation propagation between
    nearby vertices (vertex i's motion attends to its neighbours' motion)."""
    def __init__(self, n_output: int, n_inputs: int = 610,
                 n_neighbours: int = 20, n_lags: int = 5,
                 embed_dim: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_neighbours = n_neighbours
        self.n_lags = n_lags
        token_dim = 3 * n_lags                                   # 15 (3 axes x 5 lags)
        self.n_base = n_inputs - 2 * n_neighbours * token_dim    # 10

        self.global_proj = nn.Linear(self.n_base, embed_dim)
        self.token_proj  = nn.Linear(2 * token_dim, embed_dim)   # deform(15) + stress(15)
        self.pos_embed   = nn.Parameter(torch.randn(1, n_neighbours + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(embed_dim * (n_neighbours + 1), 256),
            nn.GELU(),
            nn.Linear(256, n_output)
        )

    def forward(self, x):
        B = x.shape[0]
        n, l = self.n_neighbours, self.n_lags

        base   = x[:, :self.n_base]
        deform = x[:, self.n_base:self.n_base + n * 3 * l]
        stress = x[:, self.n_base + n * 3 * l:]

        # stored as (lag, dim, neighbour) -> tokens are per-neighbour (lag, dim)
        deform = deform.reshape(B, l, 3, n).permute(0, 3, 1, 2).reshape(B, n, l * 3)
        stress = stress.reshape(B, l, 3, n).permute(0, 3, 1, 2).reshape(B, n, l * 3)

        tokens     = self.token_proj(torch.cat([deform, stress], dim=-1))  # (B, n, embed)
        global_tok = self.global_proj(base).unsqueeze(1)                   # (B, 1, embed)

        seq = torch.cat([global_tok, tokens], dim=1) + self.pos_embed      # (B, n+1, embed)
        seq = self.encoder(seq)

        return self.head(seq.reshape(B, -1))


class LiverGNN(nn.Module):
    """Graph conv net over the FEM mesh. Adjacency is derived from the
    stiffness matrix K (nonzero vertex-vertex coupling = mesh edge), so
    message passing follows the same connectivity SOFA's solver uses."""
    def __init__(self, n_output: int, n_inputs: int,
                 adjacency: torch.Tensor, vertex_rest_pos: torch.Tensor,
                 hidden: int = 128, n_layers: int = 3):
        super().__init__()
        self.n_base = n_inputs - 600  # tool kinematics + dt_window (10)
        n_nodes = vertex_rest_pos.shape[0]
        assert n_output == n_nodes * 3

        self.register_buffer('adj', adjacency)             # (n_nodes, n_nodes)
        self.register_buffer('rest_pos', vertex_rest_pos)  # (n_nodes, 3)

        self.global_proj = nn.Linear(self.n_base, hidden)
        self.pos_proj    = nn.Linear(6, hidden)  # [rest_pos, rest_pos - tool_pos]

        self.neigh = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.self_ = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

        self.head = nn.Linear(hidden, 3)
        self.act  = nn.GELU()

    def forward(self, x):
        B = x.shape[0]
        n_nodes = self.rest_pos.shape[0]

        base     = x[:, :self.n_base]
        tool_pos = x[:, :3]   # normalized tool position (approximate spatial cue)

        g = self.global_proj(base).unsqueeze(1).expand(-1, n_nodes, -1)

        rel_pos = self.rest_pos.unsqueeze(0) - tool_pos.unsqueeze(1)  # (B, n_nodes, 3)
        rest    = self.rest_pos.unsqueeze(0).expand(B, -1, -1)
        p = self.pos_proj(torch.cat([rest, rel_pos], dim=-1))

        h = self.act(g + p)  # (B, n_nodes, hidden)

        for neigh, self_lin, norm in zip(self.neigh, self.self_, self.norms):
            agg = torch.einsum('ij,bjh->bih', self.adj, h)
            h_new = neigh(agg) + self_lin(h)
            h = self.act(norm(h_new + h))

        return self.head(h).reshape(B, -1)


def physics_loss(model: LiverPINN,
                 tool_inputs: torch.Tensor,
                 u_prev: torch.Tensor,
                 F_true: torch.Tensor,
                 K_tensor: torch.Tensor,
                 fixed_indices: list = [3, 39, 64],
                 n_vertices: int = 181) -> torch.Tensor:
    """
    Physics loss using the FEM stiffness matrix K and applied forces F.
    
    Args:
        model: Your LiverPINN network
        tool_inputs: Network inputs for the current frame
        u_prev: Absolute deformation from the PREVIOUS frame (batch, 543)
        F_true: Measured force vector for the CURRENT frame (batch, 543)
        K_tensor: The verified stiffness matrix from physics.txt (543, 543)
    """
    # 1. Forward pass to get delta deformation
    delta_u_pred = model(tool_inputs)  # (batch, 543)
    
    # 2. Reconstruct total absolute deformation
    u_total = u_prev + delta_u_pred  # (batch, 543)

    # ── CHANGE: True FEM Physics Residual (Replaces old smoothness loss) ──
    # Ku product: batch matrix multiplication via u_total @ K^T
    Ku = torch.matmul(u_total, K_tensor.t())  # (batch, 543)
    
    # Structural residual: r = Ku - F
    residual = Ku - F_true
    fem_physics_loss = torch.mean(residual ** 2)

    # ── KEEP: Rest state penalty (Forces zero input to give zero output) ───
    zero_input = torch.zeros_like(tool_inputs)
    zero_deform = model(zero_input)
    rest_loss = torch.mean(zero_deform ** 2)

    # ── KEEP: Boundary condition (Fixed nodes shouldn't move) ────────────────
    u_reshaped = delta_u_pred.reshape(len(delta_u_pred), n_vertices, 3)
    bc_loss = torch.tensor(0.0, device=tool_inputs.device)
    for vi in fixed_indices:
        bc_loss = bc_loss + torch.mean(u_reshaped[:, vi, :] ** 2)
    bc_loss = bc_loss / len(fixed_indices)

    # ── COMBINE: Add them up with your proposed small weight for physics ──
    lambda_phys = 1e-5  # Start small as planned!
    
    total_physics_loss = (lambda_phys * fem_physics_loss) + (0.1 * rest_loss) + (0.5 * bc_loss)

    return total_physics_loss