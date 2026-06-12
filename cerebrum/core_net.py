import numpy as np
import torch
from .pc_core import PCAreas
from .grid_head import GridHead
from .neuromod import Neuromodulator
from .plasticity import Eligibility, weight_update, precision_update, feedback_update, feedback_update_kp
from .counters import Counters
from .rng import SeededRNG
from .invariants import assert_scalar_M
from .types import Exogenous, to_tensor, safe_to

class CerebrumCore:
    def __init__(self, cfg, device='cpu', dtype=torch.float64):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        
        self.pc = PCAreas(cfg, device=device, dtype=dtype)
        self.grid = GridHead(cfg, device=device, dtype=dtype)
        self.grid.reset()
        self.nm = Neuromodulator(cfg, device=device, dtype=dtype)
        self.rng = SeededRNG(cfg.seed, device=device, dtype=dtype)
        self.counters = Counters()
        self._U = None
        self.elig = [Eligibility((cfg.dims[l+1],), cfg, device=device, dtype=dtype) for l in range(self.pc.L-1)]

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.pc.to(device, self.dtype)
        self.grid.to(device, self.dtype)
        self.nm.to(device, self.dtype)
        self.rng.to(device, self.dtype)
        if self._U is not None:
            self._U = safe_to(self._U, device, self.dtype)
        for e in self.elig:
            e.to(device, self.dtype)
        return self

    def _top_pred_from_grid(self, obs_dim):
        rec = self.grid.complete() if self.grid.store is not None else torch.zeros(obs_dim, device=self.device, dtype=self.dtype)
        if self._U is None:
            rng = np.random.default_rng(self.cfg.seed+7)
            U_np = 0.1*rng.standard_normal((self.cfg.dims[-1], obs_dim))
            self._U = torch.tensor(U_np, device=self.device, dtype=self.dtype)
        self.counters.record_global_infer_vectors(k=1, width=self.cfg.dims[-1])
        return self._U @ rec

    def move(self, action: Exogenous):
        self.grid.transition(action)

    def observe_and_learn(self, obs, reward):
        obs_t = to_tensor(obs, self.device, self.dtype)
        self.grid.bind(obs_t, M=1.0)
        top_pred = self._top_pred_from_grid(obs_t.numel())
        
        T = self.nm.temperature(0.0)
        for _ in range(self.cfg.n_settle):
            self.pc.settle_step(self.rng, T=T, clamp_bottom=obs_t, top_pred=top_pred,
                                counters=self.counters)
        self.pc.compute_errors(top_pred=top_pred)
        
        M = self.nm.update(reward)
        assert_scalar_M(M)
        self.counters.record_global_learn(1)
        
        with torch.no_grad():
            for l in range(self.pc.L-1):
                self.elig[l].step(a_pre=self.pc.x[l+1])
                eta_w = self.cfg.eta_w/max(self.cfg.tau_w, 1e-6)
                dW = weight_update(M=M, theta=torch.ones_like(self.pc.W[l]),
                                   Pi_post=self.pc.Pi[l], eps_post=self.pc.eps[l],
                                   elig=self.elig[l].value, eta=eta_w)
                if self.cfg.align_feedback:
                    self.pc.W[l] += dW - self.cfg.lam_kp*self.pc.W[l]
                    self.pc.B[l] += feedback_update_kp(self.pc.B[l], M=M, Pi_post=self.pc.Pi[l],
                                       eps_post=self.pc.eps[l], elig=self.elig[l].value,
                                       eta=eta_w, lam_kp=self.cfg.lam_kp)
                else:
                    self.pc.W[l] += dW
                    self.pc.B[l] += (1.0/max(self.cfg.tau_b, 1e-6))*feedback_update(self.pc.B[l],
                                       a_up=self.pc.x[l+1], eps=self.pc.eps[l], cfg=self.cfg)
                self.pc.Pi[l] = precision_update(self.pc.Pi[l], eps_sq=self.pc.eps[l]**2, cfg=self.cfg)
        return M

    def predict_obs_here(self, obs_dim):
        return self.grid.complete() if self.grid.store is not None else torch.zeros(obs_dim, device=self.device, dtype=self.dtype)
