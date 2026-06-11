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

    def _bottomup_scale_top(self):
        """L2 scale of the BOTTOM-UP reconstruction signal driving the TOP area, used as the
        reference against which an external top-down prediction is precision-balanced.

        The top area is pulled UP from below by the feedback term B[L-2] @ (f' * Pi*eps) (exactly
        the term added to its drift in settle_step) — this is the obs-driven 'bottom-up signal scale'
        the mission asks the grid top-down to be weighted COMPARABLY to. If there is no obs error yet
        (early settling), fall back to the current top-area activity ||x[top]||. This is LOCAL: it
        reads only this area's own state plus its single feedback synapse, no global objective."""
        if self.L < 2:
            return float(np.linalg.norm(self.x[-1]))
        fprime = g_deriv(self.W[self.L-2] @ self.x[self.L-1])
        fb = self.B[self.L-2] @ (fprime * (self.Pi[self.L-2] * self.eps[self.L-2]))
        s = float(np.linalg.norm(fb))
        if s == 0.0:
            s = float(np.linalg.norm(self.x[-1]))
        return s

    def _balanced_top_pred(self, top_pred):
        """Gain-normalize an EXTERNAL top-down prediction to the top area's bottom-up signal scale.
        OPT-IN via cfg.balance_grid_precision; default OFF returns top_pred unchanged (bit-identical).

        When the prediction's norm exceeds the bottom-up reference, scale it DOWN so the two
        top-down/bottom-up influences pull the top area COMPARABLY (ratio = grid_precision_ref).
        A prediction already at or below the reference is left untouched (scale clamped to <=1),
        so this never amplifies — it is a pure precision/gain down-weight on a dominating prediction."""
        if top_pred is None or not getattr(self.cfg, "balance_grid_precision", False):
            return top_pred
        pnorm = float(np.linalg.norm(top_pred))
        if pnorm == 0.0:
            return top_pred
        ref = getattr(self.cfg, "grid_precision_ref", 1.0) * self._bottomup_scale_top()
        scale = min(1.0, ref / pnorm)
        return top_pred * scale

    def predict(self, l, top_pred=None):
        """top-down prediction of area l."""
        if l < self.L-1:
            return g_act(self.W[l] @ self.x[l+1])
        if top_pred is None:
            return np.zeros_like(self.x[l])
        return self._balanced_top_pred(top_pred)

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
            if l >= 1 and self.cfg.pc_sparsity_threshold > 0.0:
                new_x[l] = np.where(np.abs(new_x[l]) < self.cfg.pc_sparsity_threshold, 0.0, new_x[l])
        if counters is not None:
            dense_ops = 0
            dyn_ops = 0
            for l in range(self.L - 1):
                d_l = self.cfg.dims[l]
                d_l1 = self.cfg.dims[l+1]
                dense_ops += 2 * d_l * d_l1
                active_x = int(np.sum(np.abs(self.x[l+1]) > 1e-6))
                dyn_ops += active_x * d_l
                active_eps = int(np.sum(np.abs(self.eps[l]) > 1e-6))
                dyn_ops += active_eps * d_l1
            counters.record_synaptic_ops(dense_ops, dyn_ops)
        self.x = new_x
        if counters is not None:
            for xl in self.x[1:]: counters.record_activity(xl)
