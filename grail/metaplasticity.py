import numpy as np


class MetaplasticFuse:
    """Per-synapse surprise-gated plasticity permission. Reuses the SAME Pi, eps, eligibility that
    drive inference (NO Fisher pass, NO task-boundary, NO stored anchor weights). Low surprise builds
    a consolidation reserve c -> theta->0 (frozen, protects prior tasks); high surprise erodes c ->
    theta->1 (labile, learn-on-surprise). theta multiplies the four-factor weight update."""
    def __init__(self, shape, cfg):
        self.c = np.zeros(shape)            # consolidation reserve in [0, c_max]
        self.S_bar = np.zeros(shape)        # per-synapse surprise baseline (EMA)
        self.cfg = cfg

    def _raw_surprise(self, Pi_post, eps_post, elig):
        # S_raw_ij = |Pi_i * eps_i * e_j|  (precision-weighted error-eligibility magnitude; local)
        return np.abs((Pi_post * eps_post)[:, None] * elig[None, :])

    def update(self, Pi_post, eps_post, elig):
        S_raw = self._raw_surprise(Pi_post, eps_post, elig)
        S = S_raw - self.S_bar                       # surprise relative to the (pre-update) baseline
        # ONE clear drive each: a synapse in the PREDICTIVE regime — current surprise at or below
        # its own running baseline, including the perfectly-quiet case S_raw==S_bar==0 — builds the
        # reserve; surprise that EXCEEDS the baseline erodes it, graded by how far (learn-on-surprise).
        predictive = (S_raw <= self.S_bar).astype(float)            # [S]_- regime indicator: build c
        surprising = np.maximum(S, 0.0)                             # [S]_+ magnitude: erode c
        dc = self.cfg.alpha_c*predictive*(self.cfg.c_max - self.c) - self.cfg.beta_c*surprising*self.c
        self.c = np.clip(self.c + (1.0/self.cfg.tau_c)*dc, 0.0, self.cfg.c_max)
        self.S_bar += (1.0/self.cfg.tau_S) * (S_raw - self.S_bar)   # baseline EMA (after it is used)
        theta = 1.0/(1.0 + np.exp(-self.cfg.g_theta*(S - self.c)))  # sigma(g(S - c))
        return theta
