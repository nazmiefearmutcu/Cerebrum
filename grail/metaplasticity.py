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
        S = S_raw - self.S_bar                                       # surprise relative to baseline
        self.S_bar += (1.0/self.cfg.tau_S) * (S_raw - self.S_bar)    # baseline EMA
        # [S]_+ = above-baseline surprise erodes the reserve (learn-on-surprise);
        # [S]_- = predictive (at-or-below-baseline) builds it. A quiescent synapse
        # (S_raw at/below its baseline) is maximally predictive -> consolidates.
        pos = np.maximum(S, 0.0)                                     # surprising: erode c
        neg = np.maximum(self.S_bar - S_raw, 0.0) + np.maximum(-S, 0.0)  # predictive: build c
        # a perfectly-predicted synapse (S_raw==S_bar==0) is fully predictive -> unit drive
        quiet = (S_raw <= self.S_bar).astype(float)
        neg = neg + quiet                                           # predictive-regime baseline drive
        dc = self.cfg.alpha_c*neg*(self.cfg.c_max - self.c) - self.cfg.beta_c*pos*self.c
        self.c = np.clip(self.c + (1.0/self.cfg.tau_c)*dc, 0.0, self.cfg.c_max)
        theta = 1.0/(1.0 + np.exp(-self.cfg.g_theta*(S - self.c)))   # sigma(g(S - c))
        return theta
