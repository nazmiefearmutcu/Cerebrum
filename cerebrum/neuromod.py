import numpy as np
import torch

class Neuromodulator:
    def __init__(self, cfg, b_T=0.5, a_Pi=2.0, eta0=1.0, device='cpu', dtype=torch.float64):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.r_bar = 0.0
        self.b_T, self.a_Pi, self.eta0 = b_T, a_Pi, eta0

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    def update(self, reward):
        if isinstance(reward, torch.Tensor):
            reward_val = float(reward.item())
        else:
            reward_val = float(reward)
        M = reward_val - self.r_bar
        self.r_bar += (1.0/max(self.cfg.tau_r, 1e-6)) * (reward_val - self.r_bar)  # EMA
        return torch.tensor(M, device=self.device, dtype=self.dtype)

    def temperature(self, M):
        if isinstance(M, torch.Tensor):
            return self.cfg.T_floor + self.b_T * torch.clamp(M, min=0.0)
        return self.cfg.T_floor + self.b_T * max(0.0, float(M))

    def pi_gain(self, M):
        if isinstance(M, torch.Tensor):
            clamped_input = torch.clamp(-self.a_Pi * M, min=-50.0, max=50.0)
            return 1.0 / (1.0 + torch.exp(clamped_input))
        clamped_input = np.clip(-self.a_Pi * float(M), -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(clamped_input))

    def eta(self, M):
        if isinstance(M, torch.Tensor):
            return self.eta0 * torch.clamp(M, min=0.0)
        return self.eta0 * max(0.0, float(M))

    def t_gate(self, M, eps=1e-3):
        if isinstance(M, torch.Tensor):
            return 1.0 / (torch.abs(M) + max(eps, 1e-6))
        return 1.0 / (abs(float(M)) + max(eps, 1e-6))
