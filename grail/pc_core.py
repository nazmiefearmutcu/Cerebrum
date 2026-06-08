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

    def settle_step(self, rng, T, clamp_bottom=None, top_pred=None, broadcast=None, counters=None):
        self.compute_errors(top_pred=top_pred, broadcast=broadcast)
        c = self.cfg
        new_x = [xl.copy() for xl in self.x]
        for l in range(self.L):
            if l == 0 and clamp_bottom is not None:
                new_x[0] = clamp_bottom.copy(); continue
            drift = -self.Pi[l]*self.eps[l]
            if l >= 1:  # feedback from area below via SEPARATE B[l-1] (no transpose of W)
                # f' evaluated at the area-below prediction's pre-activation W[l-1] @ x[l];
                # in the symmetric limit B=W.T this is exactly the nonlinear PC energy gradient.
                fprime = g_deriv(self.W[l-1] @ self.x[l])
                drift = drift + self.B[l-1] @ (fprime * (self.Pi[l-1]*self.eps[l-1]))
            drift = drift - c.gamma*np.sign(self.x[l])     # -dR/dx (L1 sparsity)
            step = (drift/c.tau_x)*c.dt
            noise = rng.normal(self.x[l].shape, scale=np.sqrt(2.0*T*c.dt/c.tau_x))
            new_x[l] = self.x[l] + step + noise
            if counters is not None: counters.record_synaptic_ops(self.B[l-1].size if l>=1 else 0)
        self.x = new_x
        if counters is not None:
            for xl in self.x: counters.record_activity(xl)
