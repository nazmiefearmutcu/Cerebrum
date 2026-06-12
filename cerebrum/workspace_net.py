import numpy as np
import torch
from dataclasses import replace

from .pc_core import PCAreas
from .gate import BasalGangliaGate
from .workspace import Workspace
from .neuromod import Neuromodulator
from .plasticity import Eligibility, weight_update, precision_update, feedback_update, feedback_update_kp
from .counters import Counters
from .rng import SeededRNG
from .invariants import assert_scalar_M
from .types import to_tensor

class CerebrumWorkspaceNet:
    """Stage-2 cortical workspace network: M modules compete via a stochastic gate for k slots;
    winners' content is broadcast back as top-down prediction. Routing EMERGES from the loop;
    there is no attention/mixer module."""
    def __init__(self, n_modules, k_slots, slice_dim, cfg, device='cpu', dtype=torch.float64):
        self.cfg = cfg
        self.M_ = n_modules
        self.k = k_slots
        self.device = device
        self.dtype = dtype
        
        # each module is a PCAreas whose bottom area = its input slice
        mdims = (slice_dim,) + tuple(cfg.dims[1:]) if len(cfg.dims) > 1 else (slice_dim, slice_dim)
        self.modules = [PCAreas(replace(cfg, dims=mdims, seed=cfg.seed+i), device=device, dtype=dtype) for i in range(n_modules)]
        self.content_dim = mdims[-1]
        
        self.gate = BasalGangliaGate(n_modules, k_slots, cfg, seed=cfg.seed, device=device, dtype=dtype)
        self.workspace = Workspace(k_slots, self.content_dim, device=device, dtype=dtype)
        self.nm = Neuromodulator(cfg, device=device, dtype=dtype)
        self.rng = SeededRNG(cfg.seed, device=device, dtype=dtype)
        self.counters = Counters()
        
        self.elig = [[Eligibility((m.cfg.dims[l+1],), cfg, device=device, dtype=dtype) for l in range(m.L-1)] for m in self.modules]

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        for mod in self.modules:
            mod.to(device, self.dtype)
        self.gate.to(device, self.dtype)
        self.workspace.to(device, self.dtype)
        self.nm.to(device, self.dtype)
        self.rng.to(device, self.dtype)
        for m_elig in self.elig:
            for e in m_elig:
                e.to(device, self.dtype)
        return self

    def step(self, obs_slices, reward):
        bcast = self.workspace.broadcast()                          # top-down efference copy from last step
        self.counters.record_global_infer_vectors(k=self.k, width=self.content_dim)
        
        bcast_t = to_tensor(bcast, self.device, self.dtype)
        # top-down prediction to each module's top area = a frozen projection of the broadcast
        if bcast_t.numel() >= self.content_dim:
            top_pred = bcast_t[:self.content_dim]
        else:
            top_pred = torch.zeros(self.content_dim, device=self.device, dtype=self.dtype)
            
        # 1) settle every module with the broadcast as top-down
        T = self.nm.temperature(0.0)
        err_sq = torch.zeros(self.M_, device=self.device, dtype=self.dtype)
        reads = torch.zeros((self.M_, self.content_dim), device=self.device, dtype=self.dtype)
        for m_i, mod in enumerate(self.modules):
            for _ in range(self.cfg.n_settle):
                mod.settle_step(self.rng, T=T, clamp_bottom=obs_slices[m_i],
                                top_pred=top_pred, counters=self.counters)
            mod.compute_errors(top_pred=top_pred)
            err_sq[m_i] = sum(torch.sum(e**2) for e in mod.eps)
            reads[m_i] = mod.x[-1].clone()                            # module content = top-area activity
            
        # 2) gate: bid (scalar own-error) -> stochastic one-hot select -> write -> broadcast
        pi = torch.tensor([float(torch.mean(mod.Pi[-1]).item()) for mod in self.modules], device=self.device, dtype=self.dtype)
        bids = self.gate.bid(err_sq=err_sq, pi=pi)
        T_gate = self.cfg.gate_temp if self.cfg.gate_temp > 0.0 else self.nm.t_gate(max(reward, 1e-3))
        z = self.gate.select(bids, self.rng, T_gate=T_gate)
        self.workspace.write(z, reads)
        
        # 3) learn: scalar M gates module plasticity + gate learning + homeostasis
        M = self.nm.update(reward)
        assert_scalar_M(M)
        self.counters.record_global_learn(1)
        
        with torch.no_grad():
            for m_i, mod in enumerate(self.modules):
                for l in range(mod.L-1):
                    self.elig[m_i][l].step(a_pre=mod.x[l+1])
                    eta_w = self.cfg.eta_w/max(self.cfg.tau_w, 1e-6)
                    dW = weight_update(M=M, theta=torch.ones_like(mod.W[l]), Pi_post=mod.Pi[l],
                                       eps_post=mod.eps[l], elig=self.elig[m_i][l].value, eta=eta_w)
                    if self.cfg.align_feedback:
                        mod.W[l] += dW - self.cfg.lam_kp*mod.W[l]
                        mod.B[l] += feedback_update_kp(mod.B[l], M=M, Pi_post=mod.Pi[l],
                                       eps_post=mod.eps[l], elig=self.elig[m_i][l].value,
                                       eta=eta_w, lam_kp=self.cfg.lam_kp)
                    else:
                        mod.W[l] += dW
                        mod.B[l] += (1.0/max(self.cfg.tau_b, 1e-6))*feedback_update(mod.B[l], a_up=mod.x[l+1], eps=mod.eps[l], cfg=self.cfg)
                    mod.Pi[l] = precision_update(mod.Pi[l], eps_sq=mod.eps[l]**2, cfg=self.cfg)
            self.gate.learn(M=M)
            self.gate.homeostasis(M=M)   # reward-aware homeostasis (spec FM5b)
            
        return z, M
