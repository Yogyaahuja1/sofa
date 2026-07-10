"""
Flexible attention model for membrane PINN.
Generalises LagSequenceAttentionAccelVar to variable N_NEIGHBOURS and optional stress.

Input layout (assembled by training scripts):
  [dt_cum(n_lags) | dt_pred(1) | pos_vel(6) | contact(3) | [extra_global(n_extra_global)] |
   tool_hist(9*n_lags) | nb_deform(nb_w*n_lags) |
   [nb_stress(nb_w*n_lags) | nb_strain(nb_w*n_lags)] if use_stress |
   accel(3)]
where nb_w = n_neighbours * 3.
n_extra_global: extra scalar global features inserted between contact and tool_hist
  (e.g. n_extra_global=1 for young_modulus conditioning).

Total inputs (use_stress=False): n_lags + 13 + (9 + nb_w) * n_lags
Total inputs (use_stress=True):  n_lags + 13 + (9 + 3*nb_w) * n_lags
"""
import torch
import torch.nn as nn


class MembraneAttentionNet(nn.Module):
    def __init__(self, n_output: int, n_lags: int, n_neighbours: int,
                 use_stress: bool = True, n_extra_global: int = 0,
                 embed_dim: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_lags       = n_lags
        self.n_neighbours = n_neighbours
        self.use_stress   = use_stress

        nb_w = n_neighbours * 3           # per-lag deform block width
        n_stress_blocks = 2 if use_stress else 0   # stress + strain
        self.token_dim = 9 + nb_w * (1 + n_stress_blocks)

        # Offsets into flat input vector
        # n_extra_global adds scalar conditioning features (e.g. young_modulus)
        self.n_global      = n_lags + 10 + n_extra_global  # dt_cum + dt_pred + pos_vel + contact [+ extra]
        self.th_off        = self.n_global                         # tool hist
        self.nd_off        = self.th_off + 9 * n_lags             # nb deform
        self.ns_off        = self.nd_off + nb_w * n_lags          # nb stress
        self.nr_off        = self.ns_off + (nb_w * n_lags if use_stress else 0)  # nb strain
        self.accel_off     = self.nr_off + (nb_w * n_lags if use_stress else 0)
        self.n_inputs_expected = self.accel_off + 3

        self.nb_w = nb_w

        self.token_proj  = nn.Linear(self.token_dim, embed_dim)
        self.global_proj = nn.Linear(self.n_global + 3, embed_dim)  # global + accel
        self.pos_embed   = nn.Parameter(torch.randn(1, n_lags + 1, embed_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(embed_dim * (n_lags + 1), 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, n_output)
        )

    def forward(self, x):
        B = x.shape[0]
        assert x.shape[1] == self.n_inputs_expected, \
            f"Expected {self.n_inputs_expected} inputs, got {x.shape[1]}"

        global_feat = torch.cat(
            [x[:, :self.n_global], x[:, self.accel_off:self.accel_off + 3]], dim=1
        )

        tokens = []
        for k in range(self.n_lags):
            th = x[:, self.th_off + 9 * k           : self.th_off + 9 * (k + 1)]
            nd = x[:, self.nd_off + self.nb_w * k   : self.nd_off + self.nb_w * (k + 1)]
            parts = [th, nd]
            if self.use_stress:
                ns = x[:, self.ns_off + self.nb_w * k : self.ns_off + self.nb_w * (k + 1)]
                nr = x[:, self.nr_off + self.nb_w * k : self.nr_off + self.nb_w * (k + 1)]
                parts += [ns, nr]
            tokens.append(torch.cat(parts, dim=1))

        tokens    = torch.stack(tokens, dim=1)
        tok_embed = self.token_proj(tokens)
        glb_embed = self.global_proj(global_feat).unsqueeze(1)
        seq       = torch.cat([glb_embed, tok_embed], dim=1) + self.pos_embed
        seq       = self.encoder(seq)
        return self.head(seq.reshape(B, -1))
