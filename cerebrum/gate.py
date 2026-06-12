import numpy as np
import torch
from .invariants import assert_one_hot, assert_scalar_M
from .types import to_tensor, safe_to

class BasalGangliaGate:
    """Stochastic basal-ganglia gate. Modules bid a SCALAR own-error salience for k workspace slots;
    a striatal Go/NoGo competition selects a strict one-hot winner per slot WITH noise (never argmax,
    never soft). Gate weights learn by a LOCAL three-factor rule gated by the scalar neuromodulator M.
    There is NO query-key / content-similarity term anywhere — the competition can never become attention."""
    def __init__(self, n_modules, k_slots, cfg, seed=0, device='cpu', dtype=torch.float64):
        self.M_ = n_modules
        self.k = k_slots
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        
        rng = np.random.default_rng(seed + 31)
        G_np = 0.5 + 0.1*rng.standard_normal((n_modules, k_slots))   # Go weights
        N_np = 0.1*rng.standard_normal((n_modules, k_slots))         # NoGo weights
        
        self.G = torch.tensor(G_np, device=device, dtype=dtype)
        self.N = torch.tensor(N_np, device=device, dtype=dtype)
        self.theta = torch.zeros(n_modules, device=device, dtype=dtype)  # dead-expert excitability
        self._P = None
        self._z = None
        self._bid = None

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.G = safe_to(self.G, device, self.dtype)
        self.N = safe_to(self.N, device, self.dtype)
        self.theta = safe_to(self.theta, device, self.dtype)
        if self._P is not None:
            self._P = safe_to(self._P, device, self.dtype)
        if self._z is not None:
            self._z = safe_to(self._z, device, self.dtype)
        if self._bid is not None:
            self._bid = safe_to(self._bid, device, self.dtype)
        return self

    def bid(self, err_sq, pi):
        err_sq_t = to_tensor(err_sq, self.device, self.dtype)
        if isinstance(pi, (torch.Tensor, np.ndarray)):
            pi_val = to_tensor(pi, self.device, self.dtype)
        elif isinstance(pi, (list, tuple)):
            pi_val = torch.tensor(pi, device=self.device, dtype=self.dtype)
        else:
            pi_val = float(pi)
        return pi_val * err_sq_t + self.theta


    def select(self, bids, rng, T_gate):
        bids_t = to_tensor(bids, self.device, self.dtype)
        z = torch.zeros((self.M_, self.k), device=self.device, dtype=self.dtype)
        P = torch.zeros((self.M_, self.k), device=self.device, dtype=self.dtype)
        
        if isinstance(T_gate, torch.Tensor):
            T_val = float(T_gate.item())
        else:
            T_val = float(T_gate)
            
        for j in range(self.k):
            inhib_total = torch.sum(self.N[:, j] * bids_t)
            u = self.G[:, j] * bids_t - (inhib_total - self.N[:, j] * bids_t)
            
            # Draw Gumbel noise using SeededRNG
            gumbel_noise = rng.gumbel((self.M_,))
            logits = u / max(T_val, 1e-6) + gumbel_noise
            
            # Analytical probability under policy (un-noised utility / T)
            u_scaled = u / max(T_val, 1e-6)
            ex = torch.exp(u_scaled - u_scaled.max())
            P[:, j] = ex / ex.sum()
            z[torch.argmax(logits).item(), j] = 1.0
            
        assert_one_hot(z, axis=0)
        self._P, self._z, self._bid = P, z, bids_t
        return z

    def learn(self, M, eta=None):
        assert_scalar_M(M)
        eta_val = self.cfg.eta_w if eta is None else eta
        if isinstance(M, torch.Tensor):
            M_val = safe_to(M, self.device, self.dtype)
        else:
            M_val = float(M)
            
        e = (self._z - self._P) * self._bid[:, None]
        
        with torch.no_grad():
            self.G += eta_val * M_val * e
            self.N += -eta_val * M_val * e
            if self.cfg.lam_g > 0.0:
                self.G += self.cfg.lam_g * (0.5 - self.G)
                self.N += self.cfg.lam_g * (0.0 - self.N)

    def homeostasis(self, M=None, gamma_up=0.02, gamma_dn=0.05):
        wins = torch.minimum(self._z.sum(dim=1), torch.tensor(1.0, device=self.device, dtype=self.dtype))
        if M is None:
            hog = 1.0
        else:
            assert_scalar_M(M)
            if isinstance(M, torch.Tensor):
                M_val = float(M.item())
            else:
                M_val = float(M)
            hog = 1.0 / (1.0 + np.exp(2.0 * M_val))
            
        with torch.no_grad():
            self.theta += gamma_up * (1.0 - wins) - gamma_dn * wins * hog
            # Clamp excitability values to [-2.0, 2.0] range to prevent runaway growth
            self.theta = torch.clamp(self.theta, min=-2.0, max=2.0)
