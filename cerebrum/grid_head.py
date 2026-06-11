import numpy as np
from .invariants import assert_exogenous_action

class GridHead:
    """Structured generative prior. Frozen multi-frequency grid modules; phase advanced by
    EXOGENOUS actions only. Code g_m = [cos, sin] of phase. Content store added in Task 11."""
    def __init__(self, cfg):
        self.cfg = cfg; M = cfg.grid_n_modules
        rng = np.random.default_rng(cfg.seed + 999)
        periods = cfg.grid_lambda0 * (cfg.grid_ratio ** np.arange(M))
        angles = rng.uniform(0, 2*np.pi, size=M)            # frozen orientation per module
        self.k = np.stack([(2*np.pi/periods)*np.cos(angles),
                           (2*np.pi/periods)*np.sin(angles)], axis=1)  # (M,2) frozen frequencies
        self.pos = np.zeros(2)
        self.store = None; self.obs_dim = None
    def reset(self):
        self.pos = np.zeros(2)
    def transition(self, action):
        assert_exogenous_action(action)
        self.pos = self.pos + action.value
    def encode(self):
        phase = self.k @ self.pos                            # (M,)
        return np.stack([np.cos(phase), np.sin(phase)], axis=1).reshape(-1)
    def _ensure_store(self, obs_dim):
        if self.store is None:
            self.obs_dim = obs_dim
            self.store = np.zeros((obs_dim, self.encode().size))   # M_t: (obs_dim, grid_dim)
    def bind(self, obs, M=1.0):
        obs = np.asarray(obs, float); self._ensure_store(obs.size)
        g = self.encode()
        self.store += self.cfg.grid_eta_bind * M * np.outer(obs, g)   # Hebbian outer product
    def complete(self):
        g = self.encode()
        return self.store @ g if self.store is not None else np.zeros(self.obs_dim or 1)
