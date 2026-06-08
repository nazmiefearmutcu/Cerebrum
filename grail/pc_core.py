import numpy as np
from .nonlinear import g_act, g_deriv

class PCAreas:
    """Hierarchical predictive-coding areas. x[l] predicted from x[l+1] by forward W[l].
    Feedback B[l] is a SEPARATE synapse (no weight transport). Precision Pi[l] is DIAGONAL."""
    def __init__(self, cfg):
        self.cfg = cfg
        d = cfg.dims; self.L = len(d)
        rng = np.random.default_rng(cfg.seed)
        self.x   = [np.zeros(d[l]) for l in range(self.L)]
        self.eps = [np.zeros(d[l]) for l in range(self.L)]
        self.Pi  = [np.full(d[l], cfg.Pi0) for l in range(self.L)]
        # W[l]: (d[l], d[l+1]); B[l]: (d[l+1], d[l]) separate feedback (NOT W[l].T)
        self.W = [0.1*rng.standard_normal((d[l], d[l+1])) for l in range(self.L-1)]
        self.B = [0.1*rng.standard_normal((d[l+1], d[l])) for l in range(self.L-1)]

    def predict(self, l, top_pred=None):
        """top-down prediction of area l."""
        if l < self.L-1:
            return g_act(self.W[l] @ self.x[l+1])
        return np.zeros_like(self.x[l]) if top_pred is None else top_pred

    def compute_errors(self, top_pred=None, broadcast=None):
        for l in range(self.L):
            yhat = self.predict(l, top_pred=top_pred)
            p = 0.0 if broadcast is None else broadcast[l]   # workspace efference copy (Stage 2); 0 here
            self.eps[l] = self.x[l] - yhat - p

    def energy(self):
        e = 0.0
        for l in range(self.L):
            e += 0.5*np.sum(self.Pi[l]*self.eps[l]**2) - 0.5*np.sum(np.log(self.Pi[l]))
        return e
