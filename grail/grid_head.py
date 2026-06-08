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
    def reset(self):
        self.pos = np.zeros(2)
    def transition(self, action):
        assert_exogenous_action(action)
        self.pos = self.pos + action.value
    def encode(self):
        phase = self.k @ self.pos                            # (M,)
        return np.stack([np.cos(phase), np.sin(phase)], axis=1).reshape(-1)
