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
import torch
from dataclasses import replace

from .pc_core import PCAreas
from .grid_head import GridHead
from .gate import BasalGangliaGate
from .workspace import Workspace
from .neuromod import Neuromodulator
from .metaplasticity import MetaplasticFuse
from .plasticity import Eligibility, weight_update, precision_update, feedback_update, feedback_update_kp
from .counters import Counters
from .rng import SeededRNG
from .invariants import assert_scalar_M
from .types import Exogenous, to_tensor, safe_to


class CerebrumNet:
    def __init__(self, n_modules, k_slots, slice_dim, cfg, device='cpu', dtype=torch.float64):
        import threading
        self._lock = threading.RLock()
        self.cfg = cfg
        self.M_ = n_modules
        self.k = k_slots
        self.slice_dim = slice_dim
        self.device = device
        self.dtype = dtype
        
        # each module is a PCAreas whose bottom area is its input slice (mirrors CerebrumWorkspaceNet)
        mdims = (slice_dim,) + tuple(cfg.dims[1:]) if len(cfg.dims) > 1 else (slice_dim, slice_dim)
        self.modules = [PCAreas(replace(cfg, dims=mdims, seed=cfg.seed + i), device=device, dtype=dtype) for i in range(n_modules)]
        self.content_dim = mdims[-1]
        
        # Stage-1 structured prior: ONE shared grid head + a frozen decode U into the module top area
        self.grid = GridHead(cfg, device=device, dtype=dtype)
        self.grid.reset()
        self._U = None          # lazily built frozen decode (grid completion -> top-area prediction)
        
        # Stage-2 routing + workspace
        self.gate = BasalGangliaGate(n_modules, k_slots, cfg, seed=cfg.seed, device=device, dtype=dtype)
        self.workspace = Workspace(k_slots, self.content_dim, device=device, dtype=dtype)
        self.nm = Neuromodulator(cfg, device=device, dtype=dtype)
        self.rng = SeededRNG(cfg.seed, device=device, dtype=dtype)
        self.counters = Counters()
        
        # Instantiate Hippocampus episodic memory
        from .hippocampus import Hippocampus
        self.hippocampus = Hippocampus(key_dim=self.content_dim, capacity=1000, device=device, dtype=dtype)
        
        # one eligibility trace AND one metaplastic fuse per module per forward layer
        self.elig = [[Eligibility((m.cfg.dims[l + 1],), cfg, device=device, dtype=dtype) for l in range(m.L - 1)] for m in self.modules]
        self.fuse = [[MetaplasticFuse(m.W[l].shape, cfg, device=device, dtype=dtype) for l in range(m.L - 1)] for m in self.modules]
        
        # test/inspection hooks (not load-bearing for the algorithm)
        self._force_theta = None        # if set, pins every module-layer theta to this constant
        self.last_theta = None
        self.last_top_pred = torch.zeros(self.content_dim, device=device, dtype=dtype)
        self._backend = "numpy"

    def set_backend(self, backend, device="cpu"):
        self._backend = backend
        self.device = device
        if backend == "torch":
            dtype = torch.float32 if (device == "mps" or (isinstance(device, str) and "mps" in device)) else torch.float64
            self.to(device, dtype=dtype)
        return self

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        for mod in self.modules:
            mod.to(device, self.dtype)
        self.grid.to(device, self.dtype)
        if self._U is not None:
            self._U = safe_to(self._U, device, self.dtype)
        self.gate.to(device, self.dtype)
        self.workspace.to(device, self.dtype)
        self.nm.to(device, self.dtype)
        self.rng.to(device, self.dtype)
        for m_elig in self.elig:
            for e in m_elig:
                e.to(device, self.dtype)
        for m_fuse in self.fuse:
            for f in m_fuse:
                f.to(device, self.dtype)
        self.hippocampus.to(device, self.dtype)
        if isinstance(self.last_top_pred, torch.Tensor):
            self.last_top_pred = safe_to(self.last_top_pred, device, self.dtype)
        if self.last_theta is not None:
            for m_i in range(len(self.last_theta)):
                for layer_idx in range(len(self.last_theta[m_i])):
                    if self.last_theta[m_i][layer_idx] is not None:
                        self.last_theta[m_i][layer_idx] = safe_to(self.last_theta[m_i][layer_idx], device, self.dtype)
        return self

    # ------------------------------------------------------------------ grid prior
    def _top_pred_from_grid(self, obs_dim):
        """Structural top-down prediction: frozen decode of the (path-integrated) grid completion,
        projected into the module top-area dimension. Same pattern as CerebrumCore."""
        rec = self.grid.complete() if self.grid.store is not None else torch.zeros(obs_dim, device=self.device, dtype=self.dtype)
        if self._U is None:
            # Maintain seed parity using NumPy RNG
            rng = np.random.default_rng(self.cfg.seed + 7)
            U_np = 0.1 * rng.standard_normal((self.content_dim, obs_dim))
            self._U = torch.tensor(U_np, device=self.device, dtype=self.dtype)
        self.counters.record_global_infer_vectors(k=1, width=self.content_dim)  # broadcast to top area
        return self._U @ rec

    def _broadcast_for_module(self, mod, wksp):
        """Build the per-area efference-copy structure PCAreas.settle_step expects (broadcast[l]).
        The workspace broadcast enters ONLY the bottom area as a prediction term; other areas get 0.
        This NEVER feeds any weight update (it is removed before the four-factor rule runs)."""
        b = [torch.zeros(mod.cfg.dims[l], device=self.device, dtype=self.dtype) for l in range(mod.L)]
        d0 = mod.cfg.dims[0]
        wksp_t = to_tensor(wksp, self.device, self.dtype)
        p0 = torch.zeros(d0, device=self.device, dtype=self.dtype)
        n = min(d0, wksp_t.numel())
        p0[:n] = wksp_t[:n]
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
        err_sq = torch.zeros(self.M_, device=self.device, dtype=self.dtype)
        reads = torch.zeros((self.M_, self.content_dim), device=self.device, dtype=self.dtype)
        for m_i, mod in enumerate(self.modules):
            bcast = self._broadcast_for_module(mod, wksp)
            for _ in range(self.cfg.n_settle):
                mod.settle_step(self.rng, T=T, clamp_bottom=obs_slices[m_i],
                                top_pred=top_pred, broadcast=bcast, counters=self.counters)
                if learn:
                    for l in range(mod.L - 1):
                        self.elig[m_i][l].step(a_pre=mod.x[l + 1])
            mod.compute_errors(top_pred=top_pred, broadcast=bcast)
            err_sq[m_i] = sum(torch.sum(e ** 2) for e in mod.eps)
            reads[m_i] = mod.x[-1].clone()
        return err_sq, reads

    def settle_only(self, obs_slices, action: Exogenous, T=None):
        """Run grid path-integration + module settling WITHOUT any plasticity (for measurement).

        `T=None` uses the running inference temperature (>= T_floor, the learning-time regularizer);
        pass `T=0.0` for a deterministic noise-free readout that reflects the learned WEIGHTS rather
        than the settling floor (the same convention the Stage-3 measurement uses)."""
        with self._lock:
            self.grid.transition(action)
            top_pred = self._top_pred_from_grid(self.modules[0].cfg.dims[0])
            self.last_top_pred = top_pred.clone()
            wksp = self.workspace.broadcast()
            T = self.nm.temperature(0.0) if T is None else T
            return self._settle_all(obs_slices, top_pred, wksp, T)

    # ------------------------------------------------------------------ full step
    def step(self, obs_slices, action: Exogenous, reward):
        with self._lock:
            # (0) Sanitize inputs
            # Sanitize reward
            if isinstance(reward, torch.Tensor):
                if not torch.isfinite(reward).all():
                    reward = torch.where(torch.isfinite(reward), reward, torch.zeros_like(reward))
            elif isinstance(reward, np.ndarray):
                if not np.isfinite(reward).all():
                    reward = np.where(np.isfinite(reward), reward, 0.0)
            else:
                try:
                    reward_val = float(reward)
                    if np.isnan(reward_val) or np.isinf(reward_val):
                        reward = 0.0
                    else:
                        reward = reward_val
                except (ValueError, TypeError):
                    reward = 0.0

            # Sanitize action
            if isinstance(action, Exogenous):
                v = action.value
                if isinstance(v, torch.Tensor):
                    if not torch.isfinite(v).all():
                        v = torch.where(torch.isfinite(v), v, torch.zeros_like(v))
                        action = Exogenous(v)
                elif isinstance(v, np.ndarray):
                    if not np.isfinite(v).all():
                        v = np.where(np.isfinite(v), v, 0.0)
                        action = Exogenous(v)
                else:
                    v_arr = np.asarray(v)
                    if not np.isfinite(v_arr).all():
                        v_arr = np.where(np.isfinite(v_arr), v_arr, 0.0)
                        action = Exogenous(v_arr)

            # Sanitize obs_slices
            if not isinstance(obs_slices, (list, tuple)):
                raise TypeError("obs_slices must be a list or tuple of slices.")
            if len(obs_slices) != self.M_:
                raise ValueError(f"Number of observation slices ({len(obs_slices)}) must match n_modules={self.M_}")

            sanitized_obs_slices = []
            for obs in obs_slices:
                if isinstance(obs, (list, tuple)):
                    for item in obs:
                        if isinstance(item, (str, dict)):
                            raise TypeError("Observations must be numeric.")
                try:
                    if isinstance(obs, torch.Tensor):
                        if obs.dtype not in (torch.float16, torch.float32, torch.float64, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                            raise TypeError("Observations must be numeric.")
                        obs_conv = obs.to(device=self.device, dtype=self.dtype)
                    elif isinstance(obs, np.ndarray):
                        if obs.dtype.kind not in 'bifc':
                            raise TypeError("Observations must be numeric.")
                        obs_conv = torch.as_tensor(obs, device=self.device, dtype=self.dtype)
                    else:
                        arr = np.array(obs, dtype=np.float64)
                        if arr.dtype.kind not in 'bifc':
                            raise TypeError("Observations must be numeric.")
                        obs_conv = torch.as_tensor(arr, device=self.device, dtype=self.dtype)
                except (ValueError, TypeError) as e:
                    raise TypeError("Observations must be numeric.") from e

                # After converting to obs_conv:
                if obs_conv.ndim != 1:
                    raise ValueError("Each observation slice must be a 1D tensor/array.")

                if len(obs_conv) != self.slice_dim:
                    raise ValueError(f"Observation slice length must match slice_dim={self.slice_dim}")

                if not torch.isfinite(obs_conv).all():
                    obs_conv = torch.where(torch.isfinite(obs_conv), obs_conv, torch.zeros_like(obs_conv))
                sanitized_obs_slices.append(obs_conv)
            obs_slices = sanitized_obs_slices

            # (1) Stage-1: grid HEAD path-integrates on the EXOGENOUS action (BAN-5 enforced inside
            #     transition via assert_exogenous_action) -> structural top-down prediction.
            self.grid.transition(action)
            top_pred = self._top_pred_from_grid(self.modules[0].cfg.dims[0])
            self.last_top_pred = top_pred.clone()
            
            # Bind the (mean) observation into the grid content store, GATED by the scalar reward-
            # prediction-error preview M = r - r_bar (read BEFORE the EMA update below consumes it).
            M_preview = float(reward) - self.nm.r_bar
            obs_tensors = [to_tensor(o, self.device, self.dtype) for o in obs_slices]
            obs_mean = torch.stack(obs_tensors, dim=0).mean(dim=0)
            self.grid.bind(obs_mean, M=max(M_preview, 0.0))

            # (2) Stage-2: settle every module under grid top-down + previous workspace broadcast.
            wksp = self.workspace.broadcast()
            self.counters.record_global_infer_vectors(k=self.k, width=self.content_dim)
            T = self.nm.temperature(0.0)
            err_sq, reads = self._settle_all(obs_slices, top_pred, wksp, T, learn=True)

            # (3) Stage-2 routing: scalar own-error bid -> stochastic one-hot select -> one-hot write.
            pi = torch.tensor([float(torch.mean(mod.Pi[-1]).item()) for mod in self.modules], device=self.device, dtype=self.dtype)
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
            
            with torch.no_grad():
                for m_i, mod in enumerate(self.modules):
                    # Recompute errors without workspace broadcast to prevent efference copy corruption
                    mod.compute_errors(top_pred=top_pred, broadcast=None)
                    for l in range(mod.L - 1):
                        theta = self.fuse[m_i][l].update(mod.Pi[l], mod.eps[l], self.elig[m_i][l].value)
                        if self._force_theta is not None:
                            theta = torch.full_like(mod.W[l], float(self._force_theta), device=self.device, dtype=self.dtype)
                        self.last_theta[m_i][l] = theta
                        
                        dW = weight_update(M=M, theta=theta, Pi_post=mod.Pi[l],
                                                  eps_post=mod.eps[l], elig=self.elig[m_i][l].value,
                                                  eta=self.cfg.eta_w / max(self.cfg.tau_w, 1e-6))
                        if self.cfg.align_feedback:
                            mod.W[l] += dW - self.cfg.lam_kp * mod.W[l]
                            mod.B[l] += feedback_update_kp(mod.B[l], M=M, theta=theta, Pi_post=mod.Pi[l],
                                                           eps_post=mod.eps[l], elig=self.elig[m_i][l].value,
                                                           eta=self.cfg.eta_w / max(self.cfg.tau_w, 1e-6),
                                                           lam_kp=self.cfg.lam_kp)
                        else:
                            mod.W[l] += dW
                            dB = (1.0 / max(self.cfg.tau_b, 1e-6)) * feedback_update(mod.B[l], a_up=mod.x[l + 1],
                                                                                 eps=mod.eps[l], cfg=self.cfg)
                            mod.B[l] += dB
                        
                        mod.Pi[l] = precision_update(mod.Pi[l], eps_sq=mod.eps[l] ** 2, cfg=self.cfg)
                self.gate.learn(M=M)
                self.gate.homeostasis(M=M)            # reward-aware homeostasis (spec FM5b)
                
                # Write episode to Hippocampus episodic memory (one-shot RAG)
                key_vector = top_pred.clone()
                episode_value = {
                    "workspace": [s.clone().cpu().numpy() if isinstance(s, torch.Tensor) else s for s in self.workspace.slots],
                    "reward": float(reward),
                    "action_val": action.value.copy() if hasattr(action.value, 'copy') else action.value
                }
                self.hippocampus.write(key_vector, episode_value)
                
            return z, M
