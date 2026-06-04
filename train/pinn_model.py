import torch
import torch.nn as nn
import numpy as np


class LiverPINN(nn.Module):
    def __init__(self, n_output: int, n_inputs: int = 555):  # ← add n_inputs
        super().__init__()
        self.n_output = n_output

        # Fourier encoding — now maps 90D input
        self.n_fourier = 256
        self.register_buffer('B', torch.randn(n_inputs, self.n_fourier) * 2.0)

        n_in = 2 * self.n_fourier  # 128

        self.net = nn.Sequential(
            nn.Linear(n_in, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, n_output)
        )

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def fourier_encode(self, x):
        proj = 2 * np.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    # ADD in pinn_model.py forward():
    def forward(self, x):
        out = self.net(self.fourier_encode(x))
        return out


def physics_loss(model: LiverPINN,
                 tool_inputs: torch.Tensor,
                 fixed_indices: list = [3, 39, 64],
                 n_vertices: int = 181) -> torch.Tensor:  # Fix to 181
    """
    Force-free kinematics physics loss.
    """
    # ── Loss 1: Smoothness ────────────────────────────────────
    x = tool_inputs.clone().requires_grad_(True)
    u = model(x)  # (batch, n_vertices*3)

    grad_u = torch.autograd.grad(
        outputs=u.sum(),
        inputs=x,
        create_graph=True,
        retain_graph=True
    )[0]  

    smoothness = torch.mean(grad_u ** 2)

    # ── Loss 2: Rest state (zero velocity/displacement change → static) ───
    zero_input = torch.zeros_like(tool_inputs)
    zero_deform = model(zero_input)
    rest_loss = torch.mean(zero_deform ** 2)

    # ── Loss 3: Boundary condition ────────────────────────────
    u_reshaped = u.reshape(len(u), n_vertices, 3)
    bc_loss = torch.tensor(0.0, device=tool_inputs.device)
    for vi in fixed_indices:
        bc_loss = bc_loss + torch.mean(u_reshaped[:, vi, :] ** 2)
    bc_loss = bc_loss / len(fixed_indices)

    return smoothness + 0.1 * rest_loss + 0.5 * bc_loss