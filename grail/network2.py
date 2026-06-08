import numpy as np
from .pc_core import PCAreas
from .gate import BasalGangliaGate
from .workspace import Workspace
from .neuromod import Neuromodulator
from .plasticity import Eligibility, weight_update, precision_update, feedback_update, feedback_update_kp
from .counters import Counters
from .rng import SeededRNG
from .invariants import assert_scalar_M

class GRAILWorkspaceNet:
    """Stage-2 cortical workspace network: M modules compete via a stochastic gate for k slots;
    winners' content is broadcast back as top-down prediction. Routing EMERGES from the loop;
    there is no attention/mixer module."""
    def __init__(self, n_modules, k_slots, slice_dim, cfg):
        self.cfg = cfg; self.M_ = n_modules; self.k = k_slots
        # each module is a PCAreas whose bottom area = its input slice
        mdims = (slice_dim,) + tuple(cfg.dims[1:]) if len(cfg.dims) > 1 else (slice_dim, slice_dim)
        from dataclasses import replace
        self.modules = [PCAreas(replace(cfg, dims=mdims, seed=cfg.seed+i)) for i in range(n_modules)]
        self.content_dim = mdims[-1]
        self.gate = BasalGangliaGate(n_modules, k_slots, cfg, seed=cfg.seed)
        self.workspace = Workspace(k_slots, self.content_dim)
        self.nm = Neuromodulator(cfg)
        self.rng = SeededRNG(cfg.seed)
        self.counters = Counters()
        self.elig = [[Eligibility((m.cfg.dims[l+1],), cfg) for l in range(m.L-1)] for m in self.modules]

    def step(self, obs_slices, reward):
        bcast = self.workspace.broadcast()                          # top-down efference copy from last step
        self.counters.record_global_infer_vectors(k=self.k, width=self.content_dim)
        # top-down prediction to each module's top area = a frozen projection of the broadcast
        top_pred = bcast[:self.content_dim] if bcast.size >= self.content_dim else np.zeros(self.content_dim)
        # 1) settle every module with the broadcast as top-down
        T = self.nm.temperature(0.0)
        err_sq = np.zeros(self.M_); reads = np.zeros((self.M_, self.content_dim))
        for m_i, mod in enumerate(self.modules):
            for _ in range(self.cfg.n_settle):
                mod.settle_step(self.rng, T=T, clamp_bottom=obs_slices[m_i],
                                top_pred=top_pred, counters=self.counters)
            mod.compute_errors(top_pred=top_pred)
            err_sq[m_i] = sum(float(np.sum(e**2)) for e in mod.eps)
            reads[m_i] = mod.x[-1].copy()                            # module content = top-area activity
        # 2) gate: bid (scalar own-error) -> stochastic one-hot select -> write -> broadcast
        pi = np.array([float(np.mean(mod.Pi[-1])) for mod in self.modules])
        bids = self.gate.bid(err_sq=err_sq, pi=pi)
        T_gate = self.cfg.gate_temp if self.cfg.gate_temp > 0.0 else self.nm.t_gate(max(reward, 1e-3))
        z = self.gate.select(bids, self.rng, T_gate=T_gate)
        self.workspace.write(z, reads)
        # 3) learn: scalar M gates module plasticity + gate learning + homeostasis
        M = self.nm.update(reward); assert_scalar_M(M); self.counters.record_global_learn(1)
        for m_i, mod in enumerate(self.modules):
            for l in range(mod.L-1):
                self.elig[m_i][l].step(a_pre=mod.x[l+1])
                eta_w = self.cfg.eta_w/self.cfg.tau_w
                dW = weight_update(M=M, theta=np.ones_like(mod.W[l]), Pi_post=mod.Pi[l],
                                   eps_post=mod.eps[l], elig=self.elig[m_i][l].value, eta=eta_w)
                if self.cfg.align_feedback:
                    mod.W[l] += dW - self.cfg.lam_kp*mod.W[l]
                    mod.B[l] += feedback_update_kp(mod.B[l], M=M, Pi_post=mod.Pi[l],
                                   eps_post=mod.eps[l], elig=self.elig[m_i][l].value,
                                   eta=eta_w, lam_kp=self.cfg.lam_kp)
                else:
                    mod.W[l] += dW
                    mod.B[l] += (1.0/self.cfg.tau_b)*feedback_update(mod.B[l], a_up=mod.x[l+1], eps=mod.eps[l], cfg=self.cfg)
                mod.Pi[l] = precision_update(mod.Pi[l], eps_sq=mod.eps[l]**2, cfg=self.cfg)
        self.gate.learn(M=M); self.gate.homeostasis(M=M)   # reward-aware homeostasis (spec FM5b)
        return z, M
