import numpy as np
import torch
from .nonlinear import g_act, g_deriv
from .types import PyTorchListWrapper, to_tensor

def _raw_tensor(x):
    return getattr(x, "_tensor", x)

@torch.jit.script
def _jit_settle_update(x_l: torch.Tensor, Pi_l: torch.Tensor, eps_l: torch.Tensor, 
                       gamma: float, dt: float, tau_x: float, l2_decay: float, clip_val: float):
    # Elementwise operations:
    drift = -Pi_l * eps_l
    drift = drift - gamma * torch.sign(x_l)
    if l2_decay > 0.0:
        drift = drift - l2_decay * x_l
    if clip_val > 0.0:
        drift = torch.clamp(drift, -clip_val, clip_val)
    step = (drift / tau_x) * dt
    return step

class PCAreas:
    """Hierarchical predictive-coding areas. x[l] predicted from x[l+1] by forward W[l].
    Feedback B[l] is a SEPARATE synapse (no weight transport). Precision Pi[l] is DIAGONAL."""
    def __init__(self, cfg, device='cpu', dtype=torch.float64):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        d = cfg.dims; self.L = len(d)
        rng = np.random.default_rng(cfg.seed)
        
        x_tensors = [torch.zeros(d[l], device=device, dtype=dtype) for l in range(self.L)]
        eps_tensors = [torch.zeros(d[l], device=device, dtype=dtype) for l in range(self.L)]
        Pi_tensors = [torch.full((d[l],), cfg.Pi0, device=device, dtype=dtype) for l in range(self.L)]
        
        self._x = PyTorchListWrapper(x_tensors, device, dtype)
        self._eps = PyTorchListWrapper(eps_tensors, device, dtype)
        self._Pi = PyTorchListWrapper(Pi_tensors, device, dtype)
        
        # W[l]: (d[l], d[l+1]); B[l]: (d[l+1], d[l]) separate feedback (NOT W[l].T)
        # Weight initialization must use NumPy's RNG for seeds
        W_tensors = [torch.tensor(0.1*rng.standard_normal((d[l], d[l+1])), device=device, dtype=dtype) for l in range(self.L-1)]
        B_tensors = [torch.tensor(0.1*rng.standard_normal((d[l+1], d[l])), device=device, dtype=dtype) for l in range(self.L-1)]
        
        self._W = PyTorchListWrapper(W_tensors, device, dtype)
        self._B = PyTorchListWrapper(B_tensors, device, dtype)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, val):
        self._x = PyTorchListWrapper(val, self.device, self.dtype)

    @property
    def eps(self):
        return self._eps

    @eps.setter
    def eps(self, val):
        self._eps = PyTorchListWrapper(val, self.device, self.dtype)

    @property
    def Pi(self):
        return self._Pi

    @Pi.setter
    def Pi(self, val):
        self._Pi = PyTorchListWrapper(val, self.device, self.dtype)

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, val):
        self._W = PyTorchListWrapper(val, self.device, self.dtype)

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, val):
        self._B = PyTorchListWrapper(val, self.device, self.dtype)

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        self._x.to(device, self.dtype)
        self._eps.to(device, self.dtype)
        self._Pi.to(device, self.dtype)
        self._W.to(device, self.dtype)
        self._B.to(device, self.dtype)
        return self


    def _bottomup_scale_top(self):
        """L2 scale of the BOTTOM-UP reconstruction signal driving the TOP area, used as the
        reference against which an external top-down prediction is precision-balanced.
        """
        if self.L < 2:
            return torch.linalg.norm(self.x[-1])
        fprime = g_deriv(self.W[self.L-2] @ self.x[self.L-1])
        fb = self.B[self.L-2] @ (fprime * (self.Pi[self.L-2] * self.eps[self.L-2]))
        s = torch.linalg.norm(fb)
        if s == 0.0:
            s = torch.linalg.norm(self.x[-1])
        return s

    def _balanced_top_pred(self, top_pred):
        """Gain-normalize an EXTERNAL top-down prediction to the top area's bottom-up signal scale.
        OPT-IN via cfg.balance_grid_precision; default OFF returns top_pred unchanged (bit-identical).
        """
        if top_pred is None or not getattr(self.cfg, "balance_grid_precision", False):
            return top_pred
        pnorm = torch.linalg.norm(top_pred)
        if pnorm == 0.0:
            return top_pred
        ref = getattr(self.cfg, "grid_precision_ref", 1.0) * self._bottomup_scale_top()
        scale = torch.minimum(torch.tensor(1.0, device=self.device, dtype=self.dtype), ref / pnorm)
        return top_pred * scale

    def predict(self, l, top_pred=None):
        """top-down prediction of area l."""
        if l < self.L-1:
            return g_act(self.W[l] @ self.x[l+1])
        if top_pred is None:
            return torch.zeros_like(self.x[l])
        top_pred_t = to_tensor(top_pred, self.device, self.dtype)
        return self._balanced_top_pred(top_pred_t)

    def compute_errors(self, top_pred=None, broadcast=None):
        for l in range(self.L):
            yhat = self.predict(l, top_pred=top_pred)
            p = 0.0
            if broadcast is not None:
                p = to_tensor(broadcast[l], self.device, self.dtype)
            self.eps[l] = self.x[l] - yhat - p

    def energy(self):
        e = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        for l in range(self.L):
            e += 0.5*torch.sum(self.Pi[l]*self.eps[l]**2) - 0.5*torch.sum(torch.log(self.Pi[l]))
        return e


    def settle_step(self, rng, T, clamp_bottom=None, top_pred=None, broadcast=None, counters=None):
        self.compute_errors(top_pred=top_pred, broadcast=broadcast)
        c = self.cfg
        new_x = [xl.clone() for xl in self.x]
        
        if isinstance(T, torch.Tensor):
            T_val = float(T.item())
        else:
            T_val = float(T)
            
        for l in range(self.L):
            if l == 0 and clamp_bottom is not None:
                clamp_bottom_t = to_tensor(clamp_bottom, self.device, self.dtype)
                new_x[0] = clamp_bottom_t.clone()
                continue
                
            l2_decay = getattr(c, "pc_l2_decay", 0.001)
            clip_val = getattr(c, "pc_clip_value", 10.0)
            
            x_l_raw = _raw_tensor(self.x[l])
            Pi_l_raw = _raw_tensor(self.Pi[l])
            eps_l_raw = _raw_tensor(self.eps[l])
            
            if l >= 1:  # feedback from area below via SEPARATE B[l-1] (no transpose of W)
                W_prev_raw = _raw_tensor(self.W[l-1])
                x_curr_raw = _raw_tensor(self.x[l])
                B_prev_raw = _raw_tensor(self.B[l-1])
                Pi_prev_raw = _raw_tensor(self.Pi[l-1])
                eps_prev_raw = _raw_tensor(self.eps[l-1])
                
                fprime = g_deriv(W_prev_raw @ x_curr_raw)
                fb = B_prev_raw @ (fprime * (Pi_prev_raw * eps_prev_raw))
            else:
                fb = torch.zeros_like(x_l_raw)
                
            if getattr(c, "compile_modules", False):
                step = _jit_settle_update(x_l_raw, Pi_l_raw, eps_l_raw, 
                                          float(c.gamma), float(c.dt), float(c.tau_x), 
                                          float(l2_decay), float(clip_val))
                if l >= 1:
                    step = step + (fb / c.tau_x) * c.dt
            else:
                drift = -Pi_l_raw * eps_l_raw
                if l >= 1:
                    drift = drift + fb
                drift = drift - c.gamma * torch.sign(x_l_raw)
                if clip_val > 0.0:
                    drift = torch.clamp(drift, -clip_val, clip_val)
                if l2_decay > 0.0:
                    drift = drift - l2_decay * x_l_raw
                step = (drift / c.tau_x) * c.dt
                
            if hasattr(rng, "normal") and not isinstance(rng, np.random.Generator):
                noise = rng.normal(self.x[l].shape, scale=np.sqrt(2.0*T_val*c.dt/c.tau_x))
            else:
                scale_val = np.sqrt(2.0*T_val*c.dt/c.tau_x)
                noise_np = rng.normal(0.0, scale_val, size=self.x[l].shape)
                noise = torch.tensor(noise_np, device=self.device, dtype=self.dtype)
            
            with torch.no_grad():
                new_x[l] = self.x[l] + step + noise
                if l >= 1 and self.cfg.pc_sparsity_threshold > 0.0:
                    new_x[l] = torch.where(torch.abs(new_x[l]) < self.cfg.pc_sparsity_threshold, 
                                           torch.tensor(0.0, device=self.device, dtype=self.dtype), 
                                           new_x[l])
        
        if counters is not None:
            dense_ops = 0
            dyn_ops = 0
            for l in range(self.L - 1):
                d_l = self.cfg.dims[l]
                d_l1 = self.cfg.dims[l+1]
                dense_ops += 2 * d_l * d_l1
                active_x = int(torch.sum(torch.abs(self.x[l+1]) > 1e-6).item())
                dyn_ops += active_x * d_l
                active_eps = int(torch.sum(torch.abs(self.eps[l]) > 1e-6).item())
                dyn_ops += active_eps * d_l1
            counters.record_synaptic_ops(dense_ops, dyn_ops)
            
        self.x = PyTorchListWrapper(new_x, self.device, self.dtype)
        if counters is not None:
            for xl in self.x[1:]:
                counters.record_activity(xl)
