__version__ = "0.0.1"

import numpy as np
from dataclasses import dataclass, field, replace

# ==========================================
# 1. config.py
# ==========================================

@dataclass(frozen=True)
class CerebrumConfig:
    # PC hierarchy
    dims: tuple = (16, 12, 8)        # area sizes, dims[0] = observation/lowest area
    # timescales (smaller = faster)
    tau_x: float = 1.0               # activity settling
    tau_e: float = 5.0               # eligibility trace
    tau_w: float = 200.0             # weight plasticity (slow)
    tau_pi: float = 100.0            # precision learning
    tau_r: float = 50.0              # reward baseline EMA
    tau_b: float = 200.0             # feedback weight learning
    # settling
    dt: float = 0.1
    n_settle: int = 40
    T_floor: float = 0.02            # Pillar 4: noise floor > 0 forbids MAP collapse
    T0: float = 0.2                  # initial annealing temperature
    tau_anneal: float = 15.0
    # costs / rates
    gamma: float = 0.01              # activity L1 sparsity (R(x))
    eta_w: float = 0.02              # weight learning rate scale
    eta_b: float = 0.01              # feedback weight learning rate
    lam_b: float = 1e-3              # feedback weight decay
    # --- Kolen-Pollack feedback alignment (OPT-IN; default OFF = behavior unchanged) ---
    align_feedback: bool = False     # if True, B is driven toward W.T by a matched LOCAL rule
                                     # (KP): B and W receive the SAME M-gated pre*post product
                                     # (transposed) with a MATCHED decay, so (W - B.T) -> 0
                                     # WITHOUT ever reading/copying W.T (no weight transport).
    lam_kp: float = 1e-2             # matched symmetric weight decay applied to BOTH W and B
                                     # ONLY when align_feedback=True (the KP coupling term).
    # --- top-down precision balancing (OPT-IN; default OFF = behavior unchanged) ---
    balance_grid_precision: bool = False
                                     # if True, the EXTERNAL top-down prediction at the TOP area
                                     # (the grid HEAD's structural prediction in CerebrumNet/CerebrumCore)
                                     # is gain-normalized to the bottom-up activity scale of that
                                     # area BEFORE it enters the top-area error. In predictive coding
                                     # the relative pull of a top-down prediction is set by PRECISION;
                                     # the never-decayed Hebbian grid content store makes ||top_pred||
                                     # ~500x the small obs-driven latent (|x|~0.1), so its unit-precision
                                     # pull CRUSHES the obs factor code (the latent tracks grid phase,
                                     # not obs factors). This LOCAL, per-area diagonal gain rescales
                                     # the prediction so the grid top-down and the bottom-up
                                     # reconstruction signal are weighted COMPARABLY. NO global
                                     # objective, no weight transport — a pure prediction-gain op.
    grid_precision_ref: float = 1.0  # target ratio ||scaled top_pred|| / ||bottom-up activity||;
                                     # 1.0 = match the bottom-up scale exactly (only used when
                                     # balance_grid_precision=True).
    Pi0: float = 1.0                 # precision prior
    sigma0: float = 1.0              # precision floor variance
    kappa_pi: float = 1.0            # precision learning gain
    # grid HEAD
    grid_n_modules: int = 6
    grid_lambda0: float = 4.0        # base spatial period
    grid_ratio: float = 1.42         # geometric module scaling
    grid_eta_bind: float = 1.0       # content-store binding rate
    # gate / workspace (Stage 2)
    lam_g: float = 0.0        # gate Go/NoGo weight decay toward init (0 = off; >0 prevents spurious
                              # preference drift when there is no stable per-module target to learn)
    gate_temp: float = 0.0    # fixed gate selection temperature (0 = unset -> use neuromodulator 1/M);
                              # a low value lets the informative scalar bid dominate (still stochastic)
    # metaplasticity (Stage 3)
    tau_S: float = 20.0       # surprise-baseline EMA timescale
    tau_c: float = 300.0      # consolidation-reserve timescale (slow)
    alpha_c: float = 1.0      # low-surprise consolidation gain (builds c)
    beta_c: float = 1.5       # high-surprise erosion gain (frees c)
    c_max: float = 1.0        # max consolidation reserve
    g_theta: float = 4.0      # plasticity-permission sigmoid sharpness
    # misc
    pc_sparsity_threshold: float = 0.0
    seed: int = 0


# ==========================================
# 2. counters.py
# ==========================================

class Counters:
    def __init__(self):
        self.global_comm_learn = 0    # scalar M events (target O(1))
        self.global_comm_infer = 0    # broadcast vector elements (O(k * T_settle))
        self.synaptic_ops = 0
        self.dense_synaptic_ops = 0
        self.dynamic_synaptic_ops = 0
        self._active = 0; self._total = 0
    def record_global_learn(self, n=1): self.global_comm_learn += n
    def record_global_infer_vectors(self, k, width): self.global_comm_infer += k * width
    def record_synaptic_ops(self, dense, dynamic=None):
        if dynamic is None:
            dynamic = dense
        self.dense_synaptic_ops += int(dense)
        self.dynamic_synaptic_ops += int(dynamic)
        self.synaptic_ops = self.dynamic_synaptic_ops
    def record_activity(self, x, tol=1e-6):
        x = np.asarray(x); self._active += int(np.sum(np.abs(x) > tol)); self._total += x.size
    def sparsity(self):  # active fraction rho
        return self._active / self._total if self._total else 0.0
    def reset_activity(self): self._active = 0; self._total = 0


# ==========================================
# 3. types.py
# ==========================================

@dataclass(frozen=True)
class Exogenous:
    """An action/motor signal that is, by construction, NOT a function of network state.
    Only values explicitly wrapped here (from the task/environment) can drive the grid
    transition. This makes a data-dependent z_act a type error (BAN-1)."""
    value: np.ndarray
    def __post_init__(self):
        v = np.asarray(self.value, dtype=float)
        object.__setattr__(self, "value", v)


# ==========================================
# 4. invariants.py
# ==========================================

def assert_one_hot(z, axis=0, tol=1e-9):
    """Each slice along `axis` must be one-hot OR all-zero (an unfilled slot). Soft weights raise (BAN-1)."""
    z = np.asarray(z, dtype=float)
    moved = np.moveaxis(z, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        nz = col[np.abs(col) > tol]
        assert nz.size <= 1, f"slot {j} has {nz.size} nonzeros -> soft mixing, BAN-1 violation"
        if nz.size == 1:
            assert abs(nz[0] - 1.0) < 1e-6, f"slot {j} nonzero={nz[0]} != 1.0 -> not one-hot, BAN-1"

def assert_scalar_M(M):
    """Neuromodulator must be a scalar; a vector global signal is DFA (BAN-2)."""
    arr = np.asarray(M)
    assert arr.ndim == 0 or arr.size == 1, f"neuromodulator must be scalar, got shape {arr.shape} (DFA/BAN-2)"

def assert_exogenous_action(action):
    """Grid transition driver must be Exogenous (BAN-1: z_act not data-dependent)."""
    if not isinstance(action, Exogenous):
        raise TypeError("z_act must be Exogenous(...) — a data-dependent action is BAN-1 (selective-SSM)")


# ==========================================
# 5. grid_head.py
# ==========================================

class GridHead:
    """Structured generative prior. Frozen multi-frequency grid modules; phase advanced by
    EXOGENOUS actions only. Code g_m = [cos, sin] of phase. Content store added in Task 11."""
    def __init__(self, cfg):
        self.cfg = cfg; M = cfg.grid_n_modules
        rng = np.random.default_rng(cfg.seed + 999)
        periods = cfg.grid_lambda0 * (cfg.grid_ratio ** np.arange(M))
        angles = rng.uniform(0, 2*np.pi, size=M)            # frozen orientation per module
        self.k = np.stack([(2*np.pi/periods)*np.cos(angles),
                           (2*np.pi/periods)*np.sin(angles)], axis=1)  # (M,2) frozen frequencies
        self.pos = np.zeros(2)
        self.store = None; self.obs_dim = None
    def reset(self):
        self.pos = np.zeros(2)
    def transition(self, action):
        assert_exogenous_action(action)
        self.pos = self.pos + action.value
    def encode(self):
        phase = self.k @ self.pos                            # (M,)
        return np.stack([np.cos(phase), np.sin(phase)], axis=1).reshape(-1)
    def _ensure_store(self, obs_dim):
        if self.store is None:
            self.obs_dim = obs_dim
            self.store = np.zeros((obs_dim, self.encode().size))   # M_t: (obs_dim, grid_dim)
    def bind(self, obs, M=1.0):
        obs = np.asarray(obs, float); self._ensure_store(obs.size)
        g = self.encode()
        self.store += self.cfg.grid_eta_bind * M * np.outer(obs, g)   # Hebbian outer product
    def complete(self):
        g = self.encode()
        return self.store @ g if self.store is not None else np.zeros(self.obs_dim or 1)


# ==========================================
# 6. neuromod.py
# ==========================================

class Neuromodulator:
    def __init__(self, cfg, b_T=0.5, a_Pi=2.0, eta0=1.0):
        self.cfg = cfg; self.r_bar = 0.0
        self.b_T, self.a_Pi, self.eta0 = b_T, a_Pi, eta0
    def update(self, reward):
        M = float(reward) - self.r_bar
        self.r_bar += (1.0/self.cfg.tau_r) * (reward - self.r_bar)  # EMA
        return M
    def temperature(self, M):  return self.cfg.T_floor + self.b_T*max(0.0, M)
    def pi_gain(self, M):      return 1.0/(1.0+np.exp(-self.a_Pi*M))
    def eta(self, M):          return self.eta0*max(0.0, M)
    def t_gate(self, M, eps=1e-3): return 1.0/(abs(M)+eps)


# ==========================================
# 7. nonlinear.py
# ==========================================

def g_act(u):    return np.tanh(u)
def g_deriv(u):  return 1.0 - np.tanh(u)**2   # f = g_act' evaluated at the PRE-activation


# ==========================================
# 8. pc_core.py
# ==========================================

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


# ==========================================
# 9. plasticity.py
# ==========================================

class Eligibility:
    """Synapse-local presynaptic low-pass trace: tau_e de/dt = -e + a_pre (bare, no Pi inside)."""
    def __init__(self, shape, cfg):
        self.value = np.zeros(shape); self.cfg = cfg
    def step(self, a_pre):
        self.value += (1.0/self.cfg.tau_e)*(a_pre - self.value)
        return self.value

def weight_update(M, theta, Pi_post, eps_post, elig, eta):
    """Four-factor local Hebbian = eta * M * theta * (Pi_post*eps_post) outer elig.
    Equals eta * (-dF/dW) when M=theta=1 (precision-once convention). theta is (out,in)."""
    post = (Pi_post * eps_post)[:, None]          # (out,1)
    pre = elig[None, :]                           # (1,in)
    return eta * M * theta * (post @ pre)

def precision_update(Pi, eps_sq, cfg):
    """Diagonal, local-per-unit precision learning. Relaxes Pi toward its spec fixed point
    Pi -> 1/(sigma0^2 + <eps^2>) (spec eq. precision learning). kappa_pi sets the relaxation
    gain, tau_pi the timescale; sigma0 is the precision-floor variance. Local: each unit i
    only reads its own eps_i^2 and Pi_i, no cross-unit / matrix-inverse term."""
    target = 1.0 / np.maximum(cfg.sigma0**2 + eps_sq, 1e-6)   # 1/(sigma0^2 + <eps^2>)
    dPi = cfg.kappa_pi * (target - Pi)
    return Pi + (1.0/cfg.tau_pi)*dPi

def feedback_update(B, a_up, eps, cfg):
    """Local feedback-weight rule: eta_b * a_up outer eps - lam_b * B. No transpose of W is read."""
    return cfg.eta_b*np.outer(a_up, eps) - cfg.lam_b*B

def feedback_update_kp(B, M, Pi_post, eps_post, elig, eta, lam_kp):
    """Kolen-Pollack feedback-alignment rule (OPT-IN). Drives B[l] (shape (out_up, in_post.T)
    i.e. (d[l+1], d[l])) toward W[l].T using the SAME local four-factor product that updates
    W[l] -- only TRANSPOSED -- plus a MATCHED symmetric weight decay.

      dW[l] = eta * M * (Pi*eps) outer elig            - lam_kp * W[l]   (in network code)
      dB[l] = eta * M *  elig    outer (Pi*eps)         - lam_kp * B[l]   (this function)

    Because the two increments are exact transposes and the decay is matched, the coupled
    dynamics give d(W - B.T)/dt = -lam_kp (W - B.T), so B.T -> W exponentially (Kolen-Pollack).

    CRITICAL (BAN-3, weight transport): W.T is NEVER read or copied here. B is a separate
    physical array; it only ever sees the SAME LOCAL pre/post signals (M, Pi_post*eps_post,
    elig) that the forward synapse sees. Alignment is LEARNED, not transported. Scalar M only."""
    post = (Pi_post * eps_post)                     # (out_post,) == (d[l],)
    pre = elig                                       # (in_post,)  == (d[l+1],)
    # transpose of the forward outer(post, pre): give B the outer(pre, post) -> shape (d[l+1], d[l])
    return eta * M * np.outer(pre, post) - lam_kp * B


# ==========================================
# 10. rng.py
# ==========================================

class SeededRNG:
    """Centralized reproducible noise. enabled=False -> exact zeros (deterministic limit for tests)."""
    def __init__(self, seed: int = 0, enabled: bool = True):
        self._rng = np.random.default_rng(seed)
        self.enabled = enabled
    def normal(self, shape, scale: float = 1.0):
        if not self.enabled:
            return np.zeros(shape)
        return self._rng.normal(0.0, scale, size=shape)
    def gumbel(self, shape):
        if not self.enabled:
            return np.zeros(shape)
        return self._rng.gumbel(0.0, 1.0, size=shape)
    def uniform(self, shape):
        return self._rng.uniform(0.0, 1.0, size=shape)


# ==========================================
# 11. core_net.py
# ==========================================

class CerebrumCore:
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
            eta_w = self.cfg.eta_w/self.cfg.tau_w
            dW = weight_update(M=M, theta=np.ones_like(self.pc.W[l]),
                               Pi_post=self.pc.Pi[l], eps_post=self.pc.eps[l],
                               elig=self.elig[l].value, eta=eta_w)
            if self.cfg.align_feedback:
                # Kolen-Pollack: matched decay on W + transposed same product on B (-> B.T -> W).
                self.pc.W[l] += dW - self.cfg.lam_kp*self.pc.W[l]
                self.pc.B[l] += feedback_update_kp(self.pc.B[l], M=M, Pi_post=self.pc.Pi[l],
                                   eps_post=self.pc.eps[l], elig=self.elig[l].value,
                                   eta=eta_w, lam_kp=self.cfg.lam_kp)
            else:
                self.pc.W[l] += dW
                self.pc.B[l] += (1.0/self.cfg.tau_b)*feedback_update(self.pc.B[l],
                                   a_up=self.pc.x[l+1], eps=self.pc.eps[l], cfg=self.cfg)
            self.pc.Pi[l] = precision_update(self.pc.Pi[l], eps_sq=self.pc.eps[l]**2, cfg=self.cfg)
        return M

    def predict_obs_here(self, obs_dim):
        """Completion-based prediction at the current (path-integrated) location."""
        return self.grid.complete() if self.grid.store is not None else np.zeros(obs_dim)


# ==========================================
# 12. gate.py
# ==========================================

class BasalGangliaGate:
    """Stochastic basal-ganglia gate. Modules bid a SCALAR own-error salience for k workspace slots;
    a striatal Go/NoGo competition selects a strict one-hot winner per slot WITH noise (never argmax,
    never soft). Gate weights learn by a LOCAL three-factor rule gated by the scalar neuromodulator M.
    There is NO query-key / content-similarity term anywhere — the competition can never become attention."""
    def __init__(self, n_modules, k_slots, cfg, seed=0):
        self.M_ = n_modules; self.k = k_slots; self.cfg = cfg
        rng = np.random.default_rng(seed + 31)
        self.G = 0.5 + 0.1*rng.standard_normal((n_modules, k_slots))   # Go weights
        self.N = 0.1*rng.standard_normal((n_modules, k_slots))         # NoGo weights
        self.theta = np.zeros(n_modules)                               # dead-expert excitability
        self._P = None; self._z = None; self._bid = None

    def bid(self, err_sq, pi):
        return pi*np.asarray(err_sq) + self.theta                       # (M,) scalar per module

    def select(self, bids, rng, T_gate):
        bids = np.asarray(bids, float)
        z = np.zeros((self.M_, self.k)); P = np.zeros((self.M_, self.k))
        for j in range(self.k):
            inhib_total = float(np.sum(self.N[:, j]*bids))
            u = self.G[:, j]*bids - (inhib_total - self.N[:, j]*bids)   # u_mj = G b_m - sum_{m'!=m} N b_m'
            logits = u/max(T_gate,1e-6) + rng.gumbel((self.M_,))
            ex = np.exp(logits - logits.max()); P[:, j] = ex/ex.sum()
            z[int(np.argmax(logits)), j] = 1.0                          # Gumbel-argmax = exact softmax SAMPLE
        assert_one_hot(z, axis=0)
        self._P, self._z, self._bid = P, z, bids
        return z

    def learn(self, M, eta=None):
        assert_scalar_M(M)
        eta = self.cfg.eta_w if eta is None else eta
        e = (self._z - self._P) * self._bid[:, None]                    # local 3-factor eligibility
        self.G += eta*M*e
        self.N += -eta*M*e                                              # NoGo opponent (opposite sign)
        # Synaptic weight decay toward init (local homeostatic regularization). Without a STABLE
        # per-module target, reward-driven Go drift learns spurious preferences that override the
        # informative scalar bid; decay keeps the gate "trusting the bid" unless reward consistently
        # supports a preference. lam_g=0 -> off (unchanged behavior).
        if self.cfg.lam_g > 0.0:
            self.G += self.cfg.lam_g * (0.5 - self.G)                   # init Go mean
            self.N += self.cfg.lam_g * (0.0 - self.N)                   # init NoGo mean

    def homeostasis(self, M=None, gamma_up=0.02, gamma_dn=0.05):
        """Dead-expert load balancing as per-neuron metabolic homeostasis: excitability theta rises for
        modules that win nothing (anti-dead-expert) and falls for winners (anti-hog). When the scalar
        neuromodulator M is supplied the win-penalty is REWARD-AWARE — a REWARDED win (M>0, i.e. correct
        routing) is NOT treated as hogging — so homeostasis stops fighting correct routing (spec FM5b).
        M is a scalar (BAN-2 safe); M=None keeps the plain anti-hog behavior."""
        wins = np.minimum(self._z.sum(axis=1), 1.0)
        if M is None:
            hog = 1.0
        else:
            assert_scalar_M(M)
            hog = 1.0/(1.0 + np.exp(2.0*float(M)))     # M>0 (rewarded) -> ~0 penalty; M<=0 (hog) -> ~1
        self.theta += gamma_up*(1.0 - wins) - gamma_dn*wins*hog         # rises on loss, falls on un-rewarded win


# ==========================================
# 13. metaplasticity.py
# ==========================================

class MetaplasticFuse:
    """Per-synapse surprise-gated plasticity permission. Reuses the SAME Pi, eps, eligibility that
    drive inference (NO Fisher pass, NO task-boundary, NO stored anchor weights). Low surprise builds
    a consolidation reserve c -> theta->0 (frozen, protects prior tasks); high surprise erodes c ->
    theta->1 (labile, learn-on-surprise). theta multiplies the four-factor weight update."""
    def __init__(self, shape, cfg):
        self.c = np.zeros(shape)            # consolidation reserve in [0, c_max]
        self.S_bar = np.zeros(shape)        # per-synapse surprise baseline (EMA)
        self.cfg = cfg

    def _raw_surprise(self, Pi_post, eps_post, elig):
        # S_raw_ij = |Pi_i * eps_i * e_j|  (precision-weighted error-eligibility magnitude; local)
        return np.abs((Pi_post * eps_post)[:, None] * elig[None, :])

    def update(self, Pi_post, eps_post, elig):
        S_raw = self._raw_surprise(Pi_post, eps_post, elig)
        S = S_raw - self.S_bar                       # surprise relative to the (pre-update) baseline
        # ONE clear drive each: a synapse in the PREDICTIVE regime — current surprise at or below
        # its own running baseline, including the perfectly-quiet case S_raw==S_bar==0 — builds the
        # reserve; surprise that EXCEEDS the baseline erodes it, graded by how far (learn-on-surprise).
        predictive = (S_raw <= self.S_bar).astype(float)            # [S]_- regime indicator: build c
        surprising = np.maximum(S, 0.0)                             # [S]_+ magnitude: erode c
        dc = self.cfg.alpha_c*predictive*(self.cfg.c_max - self.c) - self.cfg.beta_c*surprising*self.c
        self.c = np.clip(self.c + (1.0/self.cfg.tau_c)*dc, 0.0, self.cfg.c_max)
        self.S_bar += (1.0/self.cfg.tau_S) * (S_raw - self.S_bar)   # baseline EMA (after it is used)
        theta = 1.0/(1.0 + np.exp(-self.cfg.g_theta*(S - self.c)))  # sigma(g(S - c))
        return theta


# ==========================================
# 14. workspace.py
# ==========================================

class Workspace:
    """k<<n latent slots. STRICT one-hot write (slot j content = the single winning module's read);
    soft-weighted aggregation is FORBIDDEN (BAN-1). Broadcast returns slot contents to all modules
    as a top-down prediction (efference copy)."""
    def __init__(self, k_slots, content_dim):
        self.k = k_slots; self.dim = content_dim
        self.slots = np.zeros((k_slots, content_dim))
    def write(self, z, reads):
        assert_one_hot(z, axis=0)                       # raises on soft weights
        for j in range(self.k):
            winners = np.flatnonzero(z[:, j] > 0.5)
            if winners.size:
                self.slots[j] = reads[winners[0]]        # strict one-hot read of the winner
    def broadcast(self):
        return self.slots.sum(axis=0)


# ==========================================
# 15. unified.py
# ==========================================

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


# ==========================================
# 16. workspace_net.py
# ==========================================

class CerebrumWorkspaceNet:
    """Stage-2 cortical workspace network: M modules compete via a stochastic gate for k slots;
    winners' content is broadcast back as top-down prediction. Routing EMERGES from the loop;
    there is no attention/mixer module."""
    def __init__(self, n_modules, k_slots, slice_dim, cfg):
        self.cfg = cfg; self.M_ = n_modules; self.k = k_slots
        # each module is a PCAreas whose bottom area = its input slice
        mdims = (slice_dim,) + tuple(cfg.dims[1:]) if len(cfg.dims) > 1 else (slice_dim, slice_dim)
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


# ==========================================
# 17. energy.py
# ==========================================

def spike_sparsity(eps_list, tol=1e-6):
    """Fraction of error-neuron units that are ACTIVE (|eps| > tol). Event-driven: well-predicted
    units are silent, so this falls toward a floor as the network becomes competent."""
    active = sum(int(np.sum(np.abs(e) > tol)) for e in eps_list)
    total = sum(int(e.size) for e in eps_list)
    return active / total if total else 0.0


def dynamic_synaptic_ops(net, tol=1e-6):
    """Event-driven synaptic-op count: a forward synapse computes only when its postsynaptic error
    neuron spikes. ops = sum_l (#active eps_l) * fan-in. Silent error neurons cost ~0, so the
    dynamic op count decays with competence."""
    ops = 0
    for l in range(net.L - 1):
        active = int(np.sum(np.abs(net.eps[l]) > tol))
        ops += active * net.W[l].shape[1]      # each active error unit drives its fan-in synapses
    return ops


def dynamic_energy_magnitude(net):
    """Magnitude-weighted dynamic switching-energy proxy: sum over predicted areas of (total error
    activity Σ|eps_l|) * fan-in. In graded event-driven coding the switching energy scales with total
    error activity, so this decays SMOOTHLY as the network becomes competent (eps -> 0). It is the
    robust headline energy metric; the thresholded spike count is a conservative companion."""
    e = 0.0
    for l in range(net.L - 1):
        e += float(np.sum(np.abs(net.eps[l]))) * net.W[l].shape[1]
    return e


def dense_backprop_ops(dims):
    """Dense forward+backward MAC count for a matched backprop net: every synapse computes every
    step (rho = 1), forward AND backward. The comparator CEREBRUM's event-driven sparsity undercuts."""
    fwd = sum(dims[l] * dims[l + 1] for l in range(len(dims) - 1))
    return 2 * fwd     # forward + backward dense passes


def global_comm_per_update(dims):
    """Global-communication events crossing the whole network per WEIGHT UPDATE.
    CEREBRUM: ONE scalar neuromodulator M (a single diffuse wire). Backprop: an error VECTOR at every
    layer (O(depth) vector elements that must be transported between layers)."""
    return {
        "cerebrum_learn_scalars": 1,
        "backprop_error_vector_elems": int(sum(dims[1:])),   # error vectors at each non-input layer
    }
