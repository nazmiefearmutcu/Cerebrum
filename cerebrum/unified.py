"""I5-Unified — ONE coherent network exercising all FIVE CEREBRUM pillars together.

`CerebrumNet` fuses the three staged prototypes into a single `step(obs_slices, action, reward)`:

  Stage-1  Predictive coding + structured grid prior
      - a shared grid HEAD path-integrates on the EXOGENOUS `action` and produces a top-down
        STRUCTURAL prediction (frozen decode of the completed grid code); each module is a
        hierarchical `PCAreas` (separate error neurons, separate feedback `B`, diagonal `Pi`).
  Stage-2  Basal-ganglia gate + k<<n workspace + thalamo-cortical broadcast
      - every module settles (Langevin noise, T>=T_floor) under BOTH the grid top-down AND the
        previous step's workspace broadcast (efference copy);
      - each module emits a SCALAR own-error bid; a stochastic striatal Go/NoGo competition
        selects a STRICT one-hot winner per slot; the winner's content is written one-hot and
        broadcast back next step. Routing EMERGES — there is no attention/query-key term.
  Stage-3  Surprise-gated metaplastic fuse on the module weights
      - a per-synapse `MetaplasticFuse` (reusing the SAME Pi/eps/eligibility — NO Fisher pass,
        NO anchor, NO task-boundary) produces theta in [0,1] that MULTIPLIES the four-factor
        module weight update, so consolidated synapses freeze while surprising ones stay labile.

The ONLY non-local signal crossing the whole network into any weight update is the single
SCALAR neuromodulator `M = r - r_bar`. The workspace broadcast enters ONLY inference as a
prediction, never a weight update (that would be DFA). This class composes the existing
modules with their CURRENT signatures and does NOT duplicate their internal logic.
"""
import numpy as np
from dataclasses import replace

from .pc_core import PCAreas
from .grid_head import GridHead
from .gate import BasalGangliaGate
from .workspace import Workspace
from .neuromod import Neuromodulator
from .metaplasticity import MetaplasticFuse
from .plasticity import Eligibility, weight_update, precision_update, feedback_update
from .counters import Counters
from .rng import SeededRNG
from .invariants import assert_scalar_M
from .types import Exogenous


class CerebrumNet:
    def __init__(self, n_modules, k_slots, slice_dim, cfg):
        self.cfg = cfg
        self.M_ = n_modules
        self.k = k_slots
        # each module is a PCAreas whose bottom area is its input slice (mirrors CerebrumWorkspaceNet)
        mdims = (slice_dim,) + tuple(cfg.dims[1:]) if len(cfg.dims) > 1 else (slice_dim, slice_dim)
        self.modules = [PCAreas(replace(cfg, dims=mdims, seed=cfg.seed + i)) for i in range(n_modules)]
        self.content_dim = mdims[-1]
        # Stage-1 structured prior: ONE shared grid head + a frozen decode U into the module top area
        self.grid = GridHead(cfg)
        self.grid.reset()
        self._U = None          # lazily built frozen decode (grid completion -> top-area prediction)
        # Stage-2 routing + workspace
        self.gate = BasalGangliaGate(n_modules, k_slots, cfg, seed=cfg.seed)
        self.workspace = Workspace(k_slots, self.content_dim)
        self.nm = Neuromodulator(cfg)
        self.rng = SeededRNG(cfg.seed)
        self.counters = Counters()
        # one eligibility trace AND one metaplastic fuse per module per forward layer
        self.elig = [[Eligibility((m.cfg.dims[l + 1],), cfg) for l in range(m.L - 1)] for m in self.modules]
        self.fuse = [[MetaplasticFuse(m.W[l].shape, cfg) for l in range(m.L - 1)] for m in self.modules]
        # test/inspection hooks (not load-bearing for the algorithm)
        self._force_theta = None        # if set, pins every module-layer theta to this constant
        self.last_theta = None
        self.last_top_pred = np.zeros(self.content_dim)

    # ------------------------------------------------------------------ grid prior
    def _top_pred_from_grid(self, obs_dim):
        """Structural top-down prediction: frozen decode of the (path-integrated) grid completion,
        projected into the module top-area dimension. Same pattern as CerebrumCore."""
        rec = self.grid.complete() if self.grid.store is not None else np.zeros(obs_dim)
        if self._U is None:
            rng = np.random.default_rng(self.cfg.seed + 7)
            self._U = 0.1 * rng.standard_normal((self.content_dim, obs_dim))   # frozen decode
        self.counters.record_global_infer_vectors(k=1, width=self.content_dim)  # broadcast to top area
        return self._U @ rec

    def _broadcast_for_module(self, mod, wksp):
        """Build the per-area efference-copy structure PCAreas.settle_step expects (broadcast[l]).
        The workspace broadcast enters ONLY the bottom area as a prediction term; other areas get 0.
        This NEVER feeds any weight update (it is removed before the four-factor rule runs)."""
        b = [0.0] * mod.L
        d0 = mod.cfg.dims[0]
        p0 = np.zeros(d0)
        n = min(d0, wksp.size)
        p0[:n] = wksp[:n]
        b[0] = p0
        return b

    # ------------------------------------------------------------------ inference only (no learning)
    def _settle_all(self, obs_slices, top_pred, wksp, T, learn=False):
        """Settle every module under BOTH the grid top-down and the workspace broadcast; return
        scalar own-error energy per module and the module read-out (top-area activity).

        When `learn=True` the presynaptic eligibility trace is advanced INSIDE the settle loop
        (as in the proven continual harness) so the four-factor outer product is pattern-specific
        — eligibility tracks the latent WHILE it settles to this observation. The inference-only
        path leaves eligibility untouched."""
        err_sq = np.zeros(self.M_)
        reads = np.zeros((self.M_, self.content_dim))
        for m_i, mod in enumerate(self.modules):
            bcast = self._broadcast_for_module(mod, wksp)
            for _ in range(self.cfg.n_settle):
                mod.settle_step(self.rng, T=T, clamp_bottom=obs_slices[m_i],
                                top_pred=top_pred, broadcast=bcast, counters=self.counters)
                if learn:
                    for l in range(mod.L - 1):
                        self.elig[m_i][l].step(a_pre=mod.x[l + 1])
            mod.compute_errors(top_pred=top_pred, broadcast=bcast)
            err_sq[m_i] = sum(float(np.sum(e ** 2)) for e in mod.eps)
            reads[m_i] = mod.x[-1].copy()
        return err_sq, reads

    def settle_only(self, obs_slices, action: Exogenous, T=None):
        """Run grid path-integration + module settling WITHOUT any plasticity (for measurement).

        `T=None` uses the running inference temperature (>= T_floor, the learning-time regularizer);
        pass `T=0.0` for a deterministic noise-free readout that reflects the learned WEIGHTS rather
        than the settling floor (the same convention the Stage-3 measurement uses)."""
        self.grid.transition(action)
        top_pred = self._top_pred_from_grid(self.modules[0].cfg.dims[0])
        self.last_top_pred = top_pred.copy()
        wksp = self.workspace.broadcast()
        T = self.nm.temperature(0.0) if T is None else T
        return self._settle_all(obs_slices, top_pred, wksp, T)

    # ------------------------------------------------------------------ full step
    def step(self, obs_slices, action: Exogenous, reward):
        # (1) Stage-1: grid HEAD path-integrates on the EXOGENOUS action (BAN-5 enforced inside
        #     transition via assert_exogenous_action) -> structural top-down prediction.
        self.grid.transition(action)
        top_pred = self._top_pred_from_grid(self.modules[0].cfg.dims[0])
        self.last_top_pred = top_pred.copy()
        # Bind the (mean) observation into the grid content store, GATED by the scalar reward-
        # prediction-error preview M = r - r_bar (read BEFORE the EMA update below consumes it).
        # As reward becomes predictable M -> 0, so binding STOPS (the prior stops growing once the
        # structure is captured) — this keeps the content store bounded and is the same "write only
        # on surprise" neuromorphic story as Stage-3. Without it, constant-reward streams would let
        # the Hebbian outer-product accumulate without bound and swamp the top-down prediction.
        M_preview = float(reward) - self.nm.r_bar
        obs_mean = np.mean(np.stack([np.asarray(o, float) for o in obs_slices]), axis=0)
        self.grid.bind(obs_mean, M=max(M_preview, 0.0))

        # (2) Stage-2: settle every module under grid top-down + previous workspace broadcast.
        wksp = self.workspace.broadcast()
        self.counters.record_global_infer_vectors(k=self.k, width=self.content_dim)
        T = self.nm.temperature(0.0)
        err_sq, reads = self._settle_all(obs_slices, top_pred, wksp, T, learn=True)

        # (3) Stage-2 routing: scalar own-error bid -> stochastic one-hot select -> one-hot write.
        pi = np.array([float(np.mean(mod.Pi[-1])) for mod in self.modules])
        bids = self.gate.bid(err_sq=err_sq, pi=pi)
        T_gate = self.cfg.gate_temp if self.cfg.gate_temp > 0.0 else self.nm.t_gate(max(reward, 1e-3))
        z = self.gate.select(bids, self.rng, T_gate=T_gate)
        self.workspace.write(z, reads)        # asserts one-hot inside (BAN-1)

        # (4) Learn: single SCALAR M gates everything; per-synapse metaplastic theta gates the
        #     four-factor module weight update; gate learning + reward-aware homeostasis.
        M = self.nm.update(reward)
        assert_scalar_M(M)                    # BAN-2
        self.counters.record_global_learn(1)  # O(1) scalar-M learn-time global comm
        self.last_theta = [[None] * (mod.L - 1) for mod in self.modules]
        for m_i, mod in enumerate(self.modules):
            for l in range(mod.L - 1):
                # eligibility was advanced inside the settle loop (pattern-specific presynaptic
                # low-pass); the Stage-3 fuse REUSES the same Pi/eps/eligibility — NO Fisher pass,
                # NO anchors, NO task-boundary. theta in [0,1] multiplies the four-factor update.
                theta = self.fuse[m_i][l].update(mod.Pi[l], mod.eps[l], self.elig[m_i][l].value)
                if self._force_theta is not None:
                    theta = np.full_like(mod.W[l], float(self._force_theta))
                self.last_theta[m_i][l] = theta
                mod.W[l] += weight_update(M=M, theta=theta, Pi_post=mod.Pi[l],
                                          eps_post=mod.eps[l], elig=self.elig[m_i][l].value,
                                          eta=self.cfg.eta_w / self.cfg.tau_w)
                mod.B[l] += (1.0 / self.cfg.tau_b) * feedback_update(mod.B[l], a_up=mod.x[l + 1],
                                                                     eps=mod.eps[l], cfg=self.cfg)
                mod.Pi[l] = precision_update(mod.Pi[l], eps_sq=mod.eps[l] ** 2, cfg=self.cfg)
        self.gate.learn(M=M)
        self.gate.homeostasis(M=M)            # reward-aware homeostasis (spec FM5b)
        return z, M
