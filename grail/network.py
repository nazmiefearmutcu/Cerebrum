import numpy as np
from .pc_core import PCAreas
from .grid_head import GridHead
from .neuromod import Neuromodulator
from .plasticity import Eligibility, weight_update, precision_update, feedback_update
from .counters import Counters
from .rng import SeededRNG
from .invariants import assert_scalar_M
from .types import Exogenous

class GRAILCore:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pc = PCAreas(cfg)
        self.grid = GridHead(cfg); self.grid.reset()
        self.nm = Neuromodulator(cfg)
        self.rng = SeededRNG(cfg.seed)
        self.counters = Counters()
        # decode matrix U: grid completion (obs_dim) -> top area prediction (dims[-1])
        self._U = None
        # eligibility = presynaptic low-pass trace e_{l,j} indexed by presynaptic unit j
        # (spec: tau_e e_dot = -e + a_{l+1,j}); weight_update forms (Pi*eps) outer elig.
        self.elig = [Eligibility((cfg.dims[l+1],), cfg) for l in range(self.pc.L-1)]

    def _top_pred_from_grid(self, obs_dim):
        rec = self.grid.complete() if self.grid.store is not None else np.zeros(obs_dim)
        if self._U is None:
            rng = np.random.default_rng(self.cfg.seed+7)
            self._U = 0.1*rng.standard_normal((self.cfg.dims[-1], obs_dim))  # frozen decode
        self.counters.record_global_infer_vectors(k=1, width=self.cfg.dims[-1])  # broadcast to top area
        return self._U @ rec

    def move(self, action: Exogenous):
        self.grid.transition(action)

    def observe_and_learn(self, obs, reward):
        obs = np.asarray(obs, float)
        # 1) bind sensory to current grid code (fast content store, M-gated downstream)
        self.grid.bind(obs, M=1.0)
        top_pred = self._top_pred_from_grid(obs.size)
        # 2) settle (stochastic) with obs clamped at bottom and grid prediction at top
        T = self.nm.temperature(0.0)
        for _ in range(self.cfg.n_settle):
            self.pc.settle_step(self.rng, T=T, clamp_bottom=obs, top_pred=top_pred,
                                counters=self.counters)
        self.pc.compute_errors(top_pred=top_pred)
        # 3) neuromodulator (scalar) + local plasticity
        M = self.nm.update(reward); assert_scalar_M(M)
        self.counters.record_global_learn(1)
        for l in range(self.pc.L-1):
            self.elig[l].step(a_pre=self.pc.x[l+1])
            dW = weight_update(M=M, theta=np.ones_like(self.pc.W[l]),
                               Pi_post=self.pc.Pi[l], eps_post=self.pc.eps[l],
                               elig=self.elig[l].value, eta=self.cfg.eta_w/self.cfg.tau_w)
            self.pc.W[l] += dW
            self.pc.B[l] += (1.0/self.cfg.tau_b)*feedback_update(self.pc.B[l],
                               a_up=self.pc.x[l+1], eps=self.pc.eps[l], cfg=self.cfg)
            self.pc.Pi[l] = precision_update(self.pc.Pi[l], eps_sq=self.pc.eps[l]**2, cfg=self.cfg)
        return M

    def predict_obs_here(self, obs_dim):
        """Completion-based prediction at the current (path-integrated) location."""
        return self.grid.complete() if self.grid.store is not None else np.zeros(obs_dim)
