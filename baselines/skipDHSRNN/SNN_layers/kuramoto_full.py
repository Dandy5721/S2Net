import torch
import torch.nn as nn
import math


class SakaguchiKuramoto(nn.Module):
    r"""
      dθ_i/dt = ω_i
                + (K/N) ∑_j τ_ij sin(θ_j - θ_i - α_ij)
                + κ_i (γ_i - θ_i)
    """
    def __init__(self, N, K=1.0, dt=1.0, device='cuda'):
        super().__init__()
        self.N = N
        self.K = K
        self.dt = dt
        self.device = device

        self.omega = nn.Parameter(torch.zeros(N))

        self.alpha = nn.Parameter(torch.zeros(N, N))

        self.kappa = nn.Parameter(torch.ones(N))

    def forward(self, theta_prev, gamma_t):
        """
        theta_prev : [B, N]   θ(t-1)
        gamma_t    : [B, N]   γ(t)
        """
        B, N = theta_prev.shape
        assert N == self.N

        theta_i = theta_prev.unsqueeze(2)       # [B, N, 1]
        theta_j = theta_prev.unsqueeze(1)       # [B, 1, N]

        phase_diff = theta_j - theta_i - self.alpha

        coupling = (self.K / self.N) * torch.sum(torch.sin(phase_diff), dim=-1)  # [B, N]

        theta_dot = self.omega + coupling + self.kappa * (gamma_t - theta_prev)  # [B, N]

        theta_new = theta_prev + self.dt * theta_dot
        return theta_new
