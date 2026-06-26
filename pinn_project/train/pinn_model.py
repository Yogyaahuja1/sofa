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


class LagSequenceAttentionAccel(nn.Module):
    """LagSequenceAttention + a trailing acceleration block (lag1-vs-lag3 velocity
    difference, computed in the training script and appended at indices 960-962).
    Validated via correlation test: leading corr(accel, |dF| over next 3 steps)=0.66,
    the strongest predictive feature found — should help anticipate fast transients
    before they fully show up in deformation/stress history.

    Input layout (963): same 960 as LagSequenceAttention, plus accel(3) at the tail.
    """
    def __init__(self, n_output: int, n_inputs: int = 963, n_lags: int = 5,
                 embed_dim: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        assert n_inputs == 963, "offsets below are hardcoded for the 963-dim layout"
        self.n_lags = n_lags
        self.n_global = 15
        self.tool_hist_off, self.tool_hist_w = 15, 9
        self.nb_deform_off, self.nb_w = 60, 60
        self.nb_stress_off = 360
        self.nb_strain_off = 660
        self.accel_off = 960
        token_dim = self.tool_hist_w + 3 * self.nb_w  # 189

        self.token_proj  = nn.Linear(token_dim, embed_dim)
        self.global_proj = nn.Linear(self.n_global + 3, embed_dim)  # +3 for accel
        self.pos_embed   = nn.Parameter(torch.randn(1, n_lags + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(embed_dim * (n_lags + 1), 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, n_output)
        )

    def forward(self, x):
        B = x.shape[0]
        global_feat = torch.cat([x[:, :self.n_global], x[:, self.accel_off:self.accel_off+3]], dim=1)

        tokens = []
        for k in range(self.n_lags):
            th = x[:, self.tool_hist_off + self.tool_hist_w*k : self.tool_hist_off + self.tool_hist_w*(k+1)]
            nd = x[:, self.nb_deform_off + self.nb_w*k        : self.nb_deform_off + self.nb_w*(k+1)]
            ns = x[:, self.nb_stress_off + self.nb_w*k        : self.nb_stress_off + self.nb_w*(k+1)]
            nr = x[:, self.nb_strain_off + self.nb_w*k        : self.nb_strain_off + self.nb_w*(k+1)]
            tokens.append(torch.cat([th, nd, ns, nr], dim=1))
        tokens = torch.stack(tokens, dim=1)  # (B, 5, 189)

        tok_embed  = self.token_proj(tokens)
        glob_embed = self.global_proj(global_feat).unsqueeze(1)
        seq = torch.cat([glob_embed, tok_embed], dim=1) + self.pos_embed
        seq = self.encoder(seq)
        return self.head(seq.reshape(B, -1))


class LagSequenceAttentionAccelVar(nn.Module):
    """Same architecture as LagSequenceAttentionAccel, but with offsets computed
    from n_lags instead of hardcoded for 5 — lets the lag window size be swept
    (e.g. 5 vs 8 vs 12) without editing the model class each time. Input layout
    for a given n_lags: dt_cum(n_lags) + dt_pred(1) + pos_vel(6) + contact(3)
    + tool_hist(9*n_lags) + nb_deform(60*n_lags) + nb_stress(60*n_lags)
    + nb_strain(60*n_lags) + accel(3) = 190*n_lags + 13 total."""
    def __init__(self, n_output: int, n_inputs: int, n_lags: int = 5,
                 embed_dim: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        expected = 190 * n_lags + 13
        assert n_inputs == expected, f"n_inputs={n_inputs} doesn't match n_lags={n_lags} (expected {expected})"
        self.n_lags = n_lags
        self.n_global = n_lags + 10  # dt_cum(n_lags) + dt_pred(1) + pos_vel(6) + contact(3)
        self.tool_hist_off, self.tool_hist_w = self.n_global, 9
        self.nb_deform_off, self.nb_w = self.tool_hist_off + 9 * n_lags, 60
        self.nb_stress_off = self.nb_deform_off + 60 * n_lags
        self.nb_strain_off = self.nb_stress_off + 60 * n_lags
        self.accel_off     = self.nb_strain_off + 60 * n_lags
        token_dim = self.tool_hist_w + 3 * self.nb_w  # 189, same regardless of n_lags

        self.token_proj  = nn.Linear(token_dim, embed_dim)
        self.global_proj = nn.Linear(self.n_global + 3, embed_dim)
        self.pos_embed   = nn.Parameter(torch.randn(1, n_lags + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(embed_dim * (n_lags + 1), 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, n_output)
        )

    def forward(self, x):
        B = x.shape[0]
        global_feat = torch.cat([x[:, :self.n_global], x[:, self.accel_off:self.accel_off+3]], dim=1)

        tokens = []
        for k in range(self.n_lags):
            th = x[:, self.tool_hist_off + self.tool_hist_w*k : self.tool_hist_off + self.tool_hist_w*(k+1)]
            nd = x[:, self.nb_deform_off + self.nb_w*k        : self.nb_deform_off + self.nb_w*(k+1)]
            ns = x[:, self.nb_stress_off + self.nb_w*k        : self.nb_stress_off + self.nb_w*(k+1)]
            nr = x[:, self.nb_strain_off + self.nb_w*k        : self.nb_strain_off + self.nb_w*(k+1)]
            tokens.append(torch.cat([th, nd, ns, nr], dim=1))
        tokens = torch.stack(tokens, dim=1)

        tok_embed  = self.token_proj(tokens)
        glob_embed = self.global_proj(global_feat).unsqueeze(1)
        seq = torch.cat([glob_embed, tok_embed], dim=1) + self.pos_embed
        seq = self.encoder(seq)
        return self.head(seq.reshape(B, -1))


class LagSequenceAttentionAccelExtra(nn.Module):
    """Same as LagSequenceAttentionAccelVar, plus n_extra additional scalar
    global features appended at the very end (after accel) — e.g. a longer-
    window smoothed velocity magnitude, to give the model a stable "is this a
    hold" signal without bloating the per-lag token window itself (which was
    found to hurt sharp-transition accuracy when lags alone were extended).
    Layout: ...same as Var..., accel(3), extra(n_extra). Total = 190*n_lags+13+n_extra."""
    def __init__(self, n_output: int, n_inputs: int, n_lags: int = 8, n_extra: int = 1,
                 embed_dim: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        expected = 190 * n_lags + 13 + n_extra
        assert n_inputs == expected, f"n_inputs={n_inputs} doesn't match n_lags={n_lags}, n_extra={n_extra} (expected {expected})"
        self.n_lags = n_lags
        self.n_extra = n_extra
        self.n_global = n_lags + 10
        self.tool_hist_off, self.tool_hist_w = self.n_global, 9
        self.nb_deform_off, self.nb_w = self.tool_hist_off + 9 * n_lags, 60
        self.nb_stress_off = self.nb_deform_off + 60 * n_lags
        self.nb_strain_off = self.nb_stress_off + 60 * n_lags
        self.accel_off     = self.nb_strain_off + 60 * n_lags
        self.extra_off     = self.accel_off + 3
        token_dim = self.tool_hist_w + 3 * self.nb_w

        self.token_proj  = nn.Linear(token_dim, embed_dim)
        self.global_proj = nn.Linear(self.n_global + 3 + n_extra, embed_dim)
        self.pos_embed   = nn.Parameter(torch.randn(1, n_lags + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(embed_dim * (n_lags + 1), 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, n_output)
        )

    def forward(self, x):
        B = x.shape[0]
        global_feat = torch.cat([x[:, :self.n_global],
                                  x[:, self.accel_off:self.accel_off+3],
                                  x[:, self.extra_off:self.extra_off+self.n_extra]], dim=1)

        tokens = []
        for k in range(self.n_lags):
            th = x[:, self.tool_hist_off + self.tool_hist_w*k : self.tool_hist_off + self.tool_hist_w*(k+1)]
            nd = x[:, self.nb_deform_off + self.nb_w*k        : self.nb_deform_off + self.nb_w*(k+1)]
            ns = x[:, self.nb_stress_off + self.nb_w*k        : self.nb_stress_off + self.nb_w*(k+1)]
            nr = x[:, self.nb_strain_off + self.nb_w*k        : self.nb_strain_off + self.nb_w*(k+1)]
            tokens.append(torch.cat([th, nd, ns, nr], dim=1))
        tokens = torch.stack(tokens, dim=1)

        tok_embed  = self.token_proj(tokens)
        glob_embed = self.global_proj(global_feat).unsqueeze(1)
        seq = torch.cat([glob_embed, tok_embed], dim=1) + self.pos_embed
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


class ChannelAttention(nn.Module):
    """Attention across feature-group 'channels' (tool_hist, nb_deform, nb_stress,
    nb_realstrain) instead of across time/lag — lets the model learn which feature
    TYPE to trust most per-sample (e.g. stress proxy vs deformation during a fast
    transient), complementing sequence attention's per-lag weighting."""
    def __init__(self, group_dims, embed_dim, n_heads=4):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, embed_dim) for d in group_dims])
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=n_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, groups):
        tokens = torch.stack([p(g) for p, g in zip(self.proj, groups)], dim=1)  # (B, n_groups, embed)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        return self.norm(tokens + attn_out)  # (B, n_groups, embed)


class LiverDualAttention(nn.Module):
    """Full dual attention (TS-MSDA U-Net inspired): channel attention fuses the 4
    feature groups [tool_hist, nb_deform, nb_stress, nb_realstrain] into one richer
    token PER LAG (replacing naive concat), then sequence attention attends across
    the 5 fused lag tokens (same mechanism as LagSequenceAttention, now fed richer
    per-lag tokens instead of flattened ones).

    Input layout (960) — identical to LagSequenceAttention, see that class's docstring.
    """
    def __init__(self, n_output: int, n_inputs: int = 960, n_lags: int = 5,
                 embed_dim: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        assert n_inputs == 960, "offsets below are hardcoded for the 960-dim layout"
        self.n_lags = n_lags
        self.n_global = 15
        self.tool_hist_off, self.tool_hist_w = 15, 9
        self.nb_deform_off, self.nb_w = 60, 60
        self.nb_stress_off = 360
        self.nb_strain_off = 660

        group_dims = [self.tool_hist_w, self.nb_w, self.nb_w, self.nb_w]  # 9, 60, 60, 60
        self.channel_attn = ChannelAttention(group_dims, embed_dim, n_heads)
        self.lag_pool = nn.Linear(embed_dim * 4, embed_dim)

        self.global_proj = nn.Linear(self.n_global, embed_dim)
        self.pos_embed   = nn.Parameter(torch.randn(1, n_lags + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.seq_attn = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(embed_dim * (n_lags + 1), 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, n_output)
        )

    def forward(self, x):
        B = x.shape[0]
        global_feat = x[:, :self.n_global]

        lag_tokens = []
        for k in range(self.n_lags):
            th = x[:, self.tool_hist_off + self.tool_hist_w*k : self.tool_hist_off + self.tool_hist_w*(k+1)]
            nd = x[:, self.nb_deform_off + self.nb_w*k        : self.nb_deform_off + self.nb_w*(k+1)]
            ns = x[:, self.nb_stress_off + self.nb_w*k        : self.nb_stress_off + self.nb_w*(k+1)]
            nr = x[:, self.nb_strain_off + self.nb_w*k        : self.nb_strain_off + self.nb_w*(k+1)]
            ch_tokens = self.channel_attn([th, nd, ns, nr])          # (B, 4, embed)
            lag_tokens.append(self.lag_pool(ch_tokens.reshape(B, -1)))  # (B, embed)
        lag_tokens = torch.stack(lag_tokens, dim=1)  # (B, 5, embed)

        glob_embed = self.global_proj(global_feat).unsqueeze(1)  # (B, 1, embed)
        seq = torch.cat([glob_embed, lag_tokens], dim=1) + self.pos_embed
        seq = self.seq_attn(seq)
        return self.head(seq.reshape(B, -1))


class LagSequenceAttention(nn.Module):
    """Sequence attention over the 5 lag steps (TS-MSDA U-Net inspired), replacing
    the naive flatten+concat of lag history. Each lag's slice
    [tool_hist(9) + nb_deform(60) + nb_stress(60) + nb_realstrain(60)] = 189 dims
    becomes one token; self-attention lets the model learn which lag matters most
    per-sample instead of every lag being weighted identically by a plain MLP.

    Input layout (960, matches train_pinn_force_final.py's X_combined):
      global (15): dt_cum(5) + dt_pred(1) + pos_vel(6) + contact(3)
      tool_hist (45): 5 lags x 9, starting at offset 15
      nb_deform (300): 5 lags x 60, starting at offset 60
      nb_stress (300): 5 lags x 60, starting at offset 360
      nb_realstrain (300): 5 lags x 60, starting at offset 660
    """
    def __init__(self, n_output: int, n_inputs: int = 960, n_lags: int = 5,
                 embed_dim: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        assert n_inputs == 960, "offsets below are hardcoded for the 960-dim layout"
        self.n_lags = n_lags
        self.n_global = 15
        self.tool_hist_off, self.tool_hist_w = 15, 9
        self.nb_deform_off, self.nb_w = 60, 60
        self.nb_stress_off = 360
        self.nb_strain_off = 660
        token_dim = self.tool_hist_w + 3 * self.nb_w  # 189

        self.token_proj  = nn.Linear(token_dim, embed_dim)
        self.global_proj = nn.Linear(self.n_global, embed_dim)
        self.pos_embed   = nn.Parameter(torch.randn(1, n_lags + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(embed_dim * (n_lags + 1), 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, n_output)
        )

    def forward(self, x):
        B = x.shape[0]
        global_feat = x[:, :self.n_global]

        tokens = []
        for k in range(self.n_lags):
            th = x[:, self.tool_hist_off + self.tool_hist_w*k : self.tool_hist_off + self.tool_hist_w*(k+1)]
            nd = x[:, self.nb_deform_off + self.nb_w*k        : self.nb_deform_off + self.nb_w*(k+1)]
            ns = x[:, self.nb_stress_off + self.nb_w*k        : self.nb_stress_off + self.nb_w*(k+1)]
            nr = x[:, self.nb_strain_off + self.nb_w*k        : self.nb_strain_off + self.nb_w*(k+1)]
            tokens.append(torch.cat([th, nd, ns, nr], dim=1))
        tokens = torch.stack(tokens, dim=1)  # (B, 5, 189)

        tok_embed  = self.token_proj(tokens)                      # (B, 5, embed)
        glob_embed = self.global_proj(global_feat).unsqueeze(1)   # (B, 1, embed)
        seq = torch.cat([glob_embed, tok_embed], dim=1) + self.pos_embed
        seq = self.encoder(seq)
        return self.head(seq.reshape(B, -1))


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