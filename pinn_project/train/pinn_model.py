import torch
import torch.nn as nn

# ── Input layout (built by train_liver_pinn.py) ───────────────────────────────
# Fixed-lag models (LiverSeqAttnFlex, LiverDualAttnFlex):
#   Global (n_lags+13): dt_cum(n_lags), dt_pred(1), pos_vel(6), contact(3), accel(3)
#   Per-lag (n_lags × lag_w), lag_w = 9 + K × n_nb_feat
#     Lag k: tool_pos(3), tool_vel(3), tool_force_log(3), nb_0..nb_{K-1}
#     Each nb_j: deform(3) + optional accstress(3) + strain(3)
#
# Variable-length model (LiverDualAttnVarLen) — flag: --var-window:
#   Global_base (13+): dt_pred(1), pos_vel(6), contact(3), accel(3) [+E] [+sp]
#   dt_seq (max_steps): actual time-ago in seconds for each history row
#   Lag blocks (max_steps × lag_w): same per-lag layout as above
#   mask (B, max_steps): True = padded/ignore, passed separately to forward()


class LiverSeqAttnFlex(nn.Module):
    """
    Channel-then-sequence attention over fixed N_LAGS history rows.
    Use with --arch seq_attn.
    """
    def __init__(self, n_output, n_lags, n_neighbours, n_nb_feat,
                 embed_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.n_lags = n_lags
        self.n_neighbours = n_neighbours
        self.n_nb_feat = n_nb_feat
        self.n_global = n_lags + 13
        self.lag_w = 9 + n_neighbours * n_nb_feat
        self.lag_off = self.n_global

        self.lag_proj = nn.Sequential(
            nn.Linear(self.lag_w, embed_dim), nn.LayerNorm(embed_dim), nn.GELU()
        )
        self.global_proj = nn.Sequential(
            nn.Linear(self.n_global, embed_dim), nn.LayerNorm(embed_dim), nn.GELU()
        )
        self.pos_embed = nn.Parameter(torch.randn(1, n_lags + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.seq_attn = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * (n_lags + 1), 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, n_output)
        )

    def forward(self, x):
        B = x.shape[0]
        global_feat = x[:, :self.n_global]
        lag_tokens = []
        for k in range(self.n_lags):
            off = self.lag_off + k * self.lag_w
            lag_tokens.append(self.lag_proj(x[:, off : off + self.lag_w]))
        lag_tokens = torch.stack(lag_tokens, dim=1)
        glob_embed = self.global_proj(global_feat).unsqueeze(1)
        seq = torch.cat([glob_embed, lag_tokens], dim=1) + self.pos_embed
        seq = self.seq_attn(seq)
        return self.head(seq.reshape(B, -1))


class LiverDualAttnFlex(nn.Module):
    """
    Dual spatial + temporal attention over fixed N_LAGS history rows.
    Default architecture — use with --arch dual_attn (or no flag).
      1. Spatial: for each lag, tool queries K neighbour keys/values → one token
      2. Temporal: transformer over the N_LAGS spatial-attended tokens
    """
    def __init__(self, n_output, n_lags, n_neighbours, n_nb_feat,
                 embed_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.n_lags = n_lags
        self.n_neighbours = n_neighbours
        self.n_nb_feat = n_nb_feat
        self.n_global = n_lags + 13
        self.lag_w = 9 + n_neighbours * n_nb_feat
        self.lag_off = self.n_global

        self.tool_proj = nn.Linear(9, embed_dim)
        self.nb_proj   = nn.Linear(n_nb_feat, embed_dim)
        self.spatial_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=0.1)
        self.spatial_norm = nn.LayerNorm(embed_dim)

        self.global_proj = nn.Sequential(
            nn.Linear(self.n_global, embed_dim), nn.LayerNorm(embed_dim), nn.GELU()
        )
        self.pos_embed = nn.Parameter(torch.randn(1, n_lags + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.temporal_attn = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * (n_lags + 1), 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, n_output)
        )

    def forward(self, x):
        B = x.shape[0]
        global_feat = x[:, :self.n_global]
        lag_tokens = []
        for k in range(self.n_lags):
            off = self.lag_off + k * self.lag_w
            th = x[:, off : off + 9]
            nb_flat = x[:, off + 9 : off + self.lag_w]
            nb = nb_flat.reshape(B, self.n_neighbours, self.n_nb_feat)
            q  = self.tool_proj(th).unsqueeze(1)
            kv = self.nb_proj(nb)
            attn_out, _ = self.spatial_attn(q, kv, kv)
            attended = self.spatial_norm(attn_out + q)
            lag_tokens.append(attended.squeeze(1))
        lag_tokens = torch.stack(lag_tokens, dim=1)
        glob_embed = self.global_proj(global_feat).unsqueeze(1)
        seq = torch.cat([glob_embed, lag_tokens], dim=1) + self.pos_embed
        seq = self.temporal_attn(seq)
        return self.head(seq.reshape(B, -1))


class LiverDualAttnVarLen(nn.Module):
    """
    Variable-length version: accepts ALL FEM rows from the past T seconds.
    Use with --var-window <seconds>.
      - Spatial attention per timestep: tool queries K neighbour nodes (vectorised)
      - Time encoding: actual dt (seconds ago) projected to embed — no fixed pos_embed
      - Temporal TransformerEncoder with src_key_padding_mask → ignores padded rows
      - Mean pool over real positions → fixed-size representation regardless of history length
    """
    def __init__(self, n_output, max_steps, n_neighbours, n_nb_feat, n_global_base,
                 embed_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.max_steps     = max_steps
        self.n_neighbours  = n_neighbours
        self.n_nb_feat     = n_nb_feat
        self.lag_w         = 9 + n_neighbours * n_nb_feat
        self.n_global_base = n_global_base
        self.embed_dim     = embed_dim
        self.dt_off  = n_global_base
        self.lag_off = n_global_base + max_steps

        self.tool_proj    = nn.Linear(9, embed_dim)
        self.nb_proj      = nn.Linear(n_nb_feat, embed_dim)
        self.spatial_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=0.1)
        self.spatial_norm = nn.LayerNorm(embed_dim)
        self.time_proj    = nn.Linear(1, embed_dim)
        self.global_proj  = nn.Sequential(
            nn.Linear(n_global_base, embed_dim), nn.LayerNorm(embed_dim), nn.GELU()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.temporal_attn = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, n_output)
        )

    def forward(self, x, mask=None):
        # x:    (B, n_global_base + max_steps + max_steps * lag_w)
        # mask: (B, max_steps)  True = padded position, ignored by attention
        B, T = x.shape[0], self.max_steps

        global_feat = x[:, :self.n_global_base]
        dt_seq      = x[:, self.dt_off : self.dt_off + T]               # (B, T) seconds ago

        # Extract all lag blocks and process spatial attention in one vectorised call
        lag_block = x[:, self.lag_off : self.lag_off + T * self.lag_w]
        lag_block = lag_block.reshape(B, T, self.lag_w)

        th_all  = lag_block[:, :, :9].reshape(B * T, 9)
        nb_all  = lag_block[:, :, 9:].reshape(B * T, self.n_neighbours, self.n_nb_feat)

        q_all  = self.tool_proj(th_all).unsqueeze(1)                     # (B*T, 1, E)
        kv_all = self.nb_proj(nb_all)                                     # (B*T, K, E)
        attn_out, _ = self.spatial_attn(q_all, kv_all, kv_all)           # (B*T, 1, E)
        attended = self.spatial_norm(attn_out + q_all)
        attended = attended.reshape(B, T, self.embed_dim)                 # (B, T, E)

        time_enc = self.time_proj(dt_seq.unsqueeze(-1))                   # (B, T, E)
        seq = attended + time_enc

        seq = self.temporal_attn(seq, src_key_padding_mask=mask)          # (B, T, E)

        if mask is not None:
            real   = (~mask).float().unsqueeze(-1)                        # (B, T, 1)
            pooled = (seq * real).sum(1) / real.sum(1).clamp(min=1)      # (B, E)
        else:
            pooled = seq.mean(1)

        glob_embed = self.global_proj(global_feat)                        # (B, E)
        return self.head(torch.cat([pooled, glob_embed], dim=1))
