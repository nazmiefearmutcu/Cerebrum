import numpy as np
import torch
from .invariants import assert_exogenous_action
from .types import to_tensor, safe_to

class GridHead:
    """Structured generative prior. Frozen multi-frequency grid modules; phase advanced by
    EXOGENOUS actions only. Code g_m = [cos, sin] of phase. Content store added in Task 11."""
    def __init__(self, cfg, device='cpu', dtype=torch.float64):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        M = cfg.grid_n_modules
        
        # Maintain EXACT seed parity using NumPy RNG for initial parameters
        rng = np.random.default_rng(cfg.seed + 999)
        periods = cfg.grid_lambda0 * (cfg.grid_ratio ** np.arange(M))
        angles = rng.uniform(0, 2*np.pi, size=M)            # frozen orientation per module
        
        k_np = np.stack([(2*np.pi/periods)*np.cos(angles),
                           (2*np.pi/periods)*np.sin(angles)], axis=1)  # (M,2) frozen frequencies
        
        self.k = torch.tensor(k_np, device=device, dtype=dtype)
        self._pos = torch.zeros(2, device=device, dtype=dtype)
        self.store = None
        self.obs_dim = None

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, val):
        self._pos = to_tensor(val, self.device, self.dtype)

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.k = safe_to(self.k, device, self.dtype)
        self._pos = safe_to(self._pos, device, self.dtype)
        if self.store is not None:
            self.store = safe_to(self.store, device, self.dtype)
        return self

    def reset(self):
        self._pos = torch.zeros(2, device=self.device, dtype=self.dtype)

    def transition(self, action):
        assert_exogenous_action(action)
        val = to_tensor(action.value, self.device, self.dtype)
        self._pos = self._pos + val

    def encode(self):
        phase = self.k @ self.pos                            # (M,)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        return torch.stack([cos_phase, sin_phase], dim=1).reshape(-1)

    def _ensure_store(self, obs_dim):
        if self.store is None:
            self.obs_dim = obs_dim
            self.store = torch.zeros((obs_dim, self.encode().numel()), device=self.device, dtype=self.dtype)

    def bind(self, obs, M=1.0):
        obs_t = to_tensor(obs, self.device, self.dtype)
        self._ensure_store(obs_t.numel())
        g = self.encode()
        if isinstance(M, torch.Tensor):
            M_val = safe_to(M, self.device, self.dtype)
        else:
            M_val = float(M)
        with torch.no_grad():
            self.store += self.cfg.grid_eta_bind * M_val * torch.outer(obs_t, g)

    def complete(self):
        g = self.encode()
        if self.store is not None:
            return self.store @ g
        else:
            return torch.zeros(self.obs_dim or 1, device=self.device, dtype=self.dtype)
