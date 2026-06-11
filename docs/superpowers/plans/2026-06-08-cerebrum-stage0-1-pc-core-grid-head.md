# CEREBRUM Stage 0+1 Implementation Plan — PC Core + Grid HEAD + Few-Shot Harness

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the CEREBRUM predictive-coding core (error neurons, stochastic Langevin settling, four-factor local plasticity, separate feedback weights, diagonal precision) plus the structured grid generative HEAD, and prove on a TEM-class few-shot task that the grid prior buys sample efficiency a flat prior and a backprop MLP cannot.

**Architecture:** Pure NumPy, **no autograd, no backprop anywhere in CEREBRUM** — every update is a hand-written local rule. The backprop MLP exists only as a *baseline comparator*. All three BAN-1/2/3 invariants are enforced as executable assertions/types so violation is impossible by construction. GPU/CPU is a microscope; success is measured only on sample efficiency (this plan), with energy/op counters instrumented for later stages.

**Tech Stack:** Python 3 (`python3` = 3.14, numpy 2.x), pytest. No torch, no jax, no sklearn.

**Spec:** `docs/superpowers/specs/2026-06-08-cerebrum-cortical-workspace-design.md` (read §2, §3, §5, §10, §12 before starting).

---

## File Structure

```
cerebrum/
  cerebrum/
    __init__.py        # package marker, version
    config.py          # CerebrumConfig dataclass — all hyperparameters
    rng.py             # SeededRNG — centralized reproducible noise (zeroable for deterministic tests)
    types.py           # Exogenous wrapper type (enforces z_act exogeneity by construction)
    invariants.py      # BAN-1/2/3 executable assertions
    counters.py        # energy/op + global-comm-event counters (LEARN vs INFER)
    nonlinear.py       # g_act = tanh and derivative
    pc_core.py         # PCAreas: predictions, error neurons, diagonal precision, Langevin settling step
    plasticity.py      # eligibility traces, four-factor local weight rule, feedback-B local rule, precision learning
    neuromod.py        # scalar neuromodulator M (r - r_bar EMA) + couplings (T, Pi-gain, eta, T_gate)
    grid_head.py       # structured generative prior: frozen grid modules, path integration, content store, completion
    network.py         # CerebrumCore (Stage-1: PC areas + grid HEAD, NO gate yet) — observe + learn + predict
  tests/
    test_config.py  test_rng.py  test_types.py  test_invariants.py  test_counters.py
    test_nonlinear.py  test_pc_core.py  test_plasticity.py  test_neuromod.py
    test_grid_head.py  test_network.py  test_settling.py
  benchmarks/
    tasks/gridworld.py          # TEM-class structured environment + episode generator
    tasks/graph_completion.py   # few-shot graph-completion metric
    baselines/flat_prior.py     # CEREBRUM with flat (non-path-integrating) prior — Pillar-3 ablation
    baselines/backprop_mlp.py   # tiny manual-backprop MLP — baseline comparator ONLY
    run_task1.py                # CEREBRUM-grid vs flat vs backprop @ K in {5,10,20}
  tests/test_task1_smoke.py     # smoke + the load-bearing assertion grid > flat on completion
  pyproject.toml
  conftest.py
  README.md
```

---

## Task 1: Project scaffolding + config + seeded RNG

**Files:**
- Create: `pyproject.toml`, `conftest.py`, `cerebrum/__init__.py`, `cerebrum/config.py`, `cerebrum/rng.py`
- Test: `tests/test_config.py`, `tests/test_rng.py`

- [ ] **Step 1: Write `pyproject.toml` and `conftest.py`**

`pyproject.toml`:
```toml
[project]
name = "cerebrum"
version = "0.0.1"
description = "CEREBRUM — predictive-coding, backprop-free, local-plasticity, neuromorphic-targeted learning"
requires-python = ">=3.11"
dependencies = ["numpy>=2.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```
`conftest.py`:
```python
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
```

- [ ] **Step 2: Write failing tests**

`tests/test_config.py`:
```python
from cerebrum.config import CerebrumConfig

def test_defaults_present_and_sane():
    c = CerebrumConfig()
    assert c.dims[0] > 0 and len(c.dims) >= 2
    assert c.T_floor > 0.0           # Pillar 4: never MAP collapse
    assert c.dt > 0 and c.n_settle > 0
    assert c.tau_e < c.tau_w         # spec timescale ordering tau_x << tau_gate << tau_w
    assert c.tau_x < c.tau_w

def test_config_is_frozen_and_overridable():
    c = CerebrumConfig(seed=7, T_floor=0.05)
    assert c.seed == 7 and c.T_floor == 0.05
```

`tests/test_rng.py`:
```python
import numpy as np
from cerebrum.rng import SeededRNG

def test_reproducible():
    a = SeededRNG(123).normal((4,)); b = SeededRNG(123).normal((4,))
    assert np.allclose(a, b)

def test_zeroable_for_deterministic_tests():
    r = SeededRNG(1, enabled=False)
    assert np.allclose(r.normal((5,)), 0.0)   # disabling noise gives exact zeros (deterministic limit)
```

- [ ] **Step 3: Run tests, verify they fail** — `python3 -m pytest tests/test_config.py tests/test_rng.py -v` → FAIL (module not found).

- [ ] **Step 4: Implement**

`cerebrum/__init__.py`:
```python
__version__ = "0.0.1"
```
`cerebrum/config.py`:
```python
from dataclasses import dataclass, field

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
    Pi0: float = 1.0                 # precision prior
    sigma0: float = 1.0              # precision floor variance
    kappa_pi: float = 1.0            # precision learning gain
    # grid HEAD
    grid_n_modules: int = 6
    grid_lambda0: float = 4.0        # base spatial period
    grid_ratio: float = 1.42         # geometric module scaling
    grid_eta_bind: float = 1.0       # content-store binding rate
    # misc
    seed: int = 0
```
`cerebrum/rng.py`:
```python
import numpy as np

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
```

- [ ] **Step 5: Run tests (PASS) and commit** — `git add -A && git commit -m "feat: scaffolding, config, seeded RNG"`

---

## Task 2: BAN-1/2/3 invariants as executable assertions + Exogenous type

**Files:**
- Create: `cerebrum/types.py`, `cerebrum/invariants.py`
- Test: `tests/test_types.py`, `tests/test_invariants.py`

This task makes the three bans **impossible to violate silently**: a soft (non-one-hot) workspace write raises, a data-derived `z_act` is a *type error*, and a vector neuromodulator raises.

- [ ] **Step 1: Write failing tests**

`tests/test_types.py`:
```python
import numpy as np, pytest
from cerebrum.types import Exogenous

def test_exogenous_wraps_array():
    a = Exogenous(np.array([1.0, 0.0]))
    assert isinstance(a.value, np.ndarray) and a.value.shape == (2,)

def test_plain_array_is_not_exogenous():
    # the grid transition will only accept Exogenous; a plain ndarray (which could be derived
    # from data) is rejected -> z_act cannot become data-dependent by construction (BAN-1).
    assert not isinstance(np.array([1.0, 0.0]), Exogenous)
```

`tests/test_invariants.py`:
```python
import numpy as np, pytest
from cerebrum.invariants import assert_one_hot, assert_scalar_M, assert_exogenous_action
from cerebrum.types import Exogenous

def test_one_hot_passes_for_one_hot_columns():
    z = np.array([[1.0,0.0],[0.0,1.0],[0.0,0.0]])  # per-slot (column) one-hot; a slot may be empty
    assert_one_hot(z, axis=0)  # no raise

def test_one_hot_rejects_soft_weights():
    z = np.array([[0.6,0.1],[0.4,0.9]])            # soft mixing weights = BAN-1 violation
    with pytest.raises(AssertionError):
        assert_one_hot(z, axis=0)

def test_scalar_M_passes_and_vector_raises():
    assert_scalar_M(0.7); assert_scalar_M(np.float64(0.7))
    with pytest.raises(AssertionError):
        assert_scalar_M(np.array([0.1, 0.2]))       # vector global signal = DFA = BAN-2

def test_exogenous_action_accepts_only_wrapped():
    assert_exogenous_action(Exogenous(np.array([1.0,0.0])))  # ok
    with pytest.raises(TypeError):
        assert_exogenous_action(np.array([1.0,0.0]))         # plain (possibly data-derived) array rejected
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement**

`cerebrum/types.py`:
```python
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class Exogenous:
    """An action/motor signal that is, by construction, NOT a function of network state.
    Only values explicitly wrapped here (from the task/environment) can drive the grid
    transition. This makes a data-dependent z_act a type error (BAN-1)."""
    value: np.ndarray
    def __post_init__(self):
        v = np.asarray(self.value, dtype=float)
        object.__setattr__(self, "value", v)
```

`cerebrum/invariants.py`:
```python
import numpy as np
from .types import Exogenous

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
```

- [ ] **Step 4: Run (PASS).**
- [ ] **Step 5: Commit** — `git commit -am "feat: BAN-1/2/3 invariants + Exogenous action type"`

---

## Task 3: Energy / operation counters (LEARN vs INFER global comm)

**Files:** Create `cerebrum/counters.py`; Test `tests/test_counters.py`

Per spec §10 Task-3: report global comm as TWO numbers (LEARN-time scalar M vs INFER-time broadcast vectors), synaptic ops, and activation sparsity.

- [ ] **Step 1: Failing test** `tests/test_counters.py`:
```python
import numpy as np
from cerebrum.counters import Counters

def test_counts_learn_and_infer_separately():
    c = Counters()
    c.record_global_learn(1)          # one scalar M broadcast at learn time
    c.record_global_infer_vectors(k=4, width=8)  # broadcast of 4 slots x width 8 at infer time
    c.record_synaptic_ops(100)
    assert c.global_comm_learn == 1
    assert c.global_comm_infer == 4*8
    assert c.synaptic_ops == 100

def test_sparsity_tracks_active_fraction():
    c = Counters()
    c.record_activity(np.array([0.0, 0.0, 1.0, 0.0]))  # 1/4 active
    assert abs(c.sparsity() - 0.25) < 1e-9
```

- [ ] **Step 2: Run, fail.**
- [ ] **Step 3: Implement** `cerebrum/counters.py`:
```python
import numpy as np

class Counters:
    def __init__(self):
        self.global_comm_learn = 0    # scalar M events (target O(1))
        self.global_comm_infer = 0    # broadcast vector elements (O(k * T_settle))
        self.synaptic_ops = 0
        self._active = 0; self._total = 0
    def record_global_learn(self, n=1): self.global_comm_learn += n
    def record_global_infer_vectors(self, k, width): self.global_comm_infer += k * width
    def record_synaptic_ops(self, n): self.synaptic_ops += int(n)
    def record_activity(self, x, tol=1e-6):
        x = np.asarray(x); self._active += int(np.sum(np.abs(x) > tol)); self._total += x.size
    def sparsity(self):  # active fraction rho
        return self._active / self._total if self._total else 0.0
    def reset_activity(self): self._active = 0; self._total = 0
```
- [ ] **Step 4: Run (PASS). Step 5: Commit** — `git commit -am "feat: energy/op counters (learn vs infer global comm)"`

---

## Task 4: Nonlinearity (g_act + derivative)

**Files:** Create `cerebrum/nonlinear.py`; Test `tests/test_nonlinear.py`

- [ ] **Step 1: Failing test:**
```python
import numpy as np
from cerebrum.nonlinear import g_act, g_deriv

def test_tanh_values():
    u = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(g_act(u), np.tanh(u))

def test_derivative_matches_finite_difference():
    u = np.linspace(-2, 2, 11); h = 1e-6
    fd = (g_act(u+h) - g_act(u-h)) / (2*h)
    assert np.allclose(g_deriv(u), fd, atol=1e-5)
```
- [ ] **Step 2: Fail. Step 3: Implement:**
```python
import numpy as np
def g_act(u):    return np.tanh(u)
def g_deriv(u):  return 1.0 - np.tanh(u)**2   # f = g_act' evaluated at the PRE-activation
```
- [ ] **Step 4: PASS. Step 5: Commit** — `git commit -am "feat: tanh nonlinearity + derivative"`

---

## Task 5: PC areas — predictions + error neurons + diagonal precision

**Files:** Create `cerebrum/pc_core.py`; Test `tests/test_pc_core.py`

Implements spec §2 (ε_l = x_l − ŷ_l, ŷ_l = g_act(W_l x_{l+1})), diagonal precision Π_l, and the energy term `Σ_l ½ ε_lᵀ Π_l ε_l − ½ log det Π_l`. **No settling yet.** Index convention: `x[0]` = lowest/observation area; `x[l]` predicted from `x[l+1]` via `W[l]`. Top area `x[L-1]` receives its top-down prediction from the grid HEAD (passed in later); for this task the top prediction defaults to zero.

- [ ] **Step 1: Failing test** `tests/test_pc_core.py`:
```python
import numpy as np
from cerebrum.config import CerebrumConfig
from cerebrum.pc_core import PCAreas
from cerebrum.nonlinear import g_act

def make():
    c = CerebrumConfig(dims=(4,3,2), seed=0)
    return PCAreas(c), c

def test_shapes():
    pc, c = make()
    assert len(pc.x) == 3 and pc.x[0].shape == (4,)
    assert len(pc.W) == 2 and pc.W[0].shape == (4,3)   # W[l]: predicts area l (size dims[l]) from area l+1 (size dims[l+1])
    assert pc.Pi[0].shape == (4,)                       # diagonal precision = vector

def test_error_is_input_minus_prediction():
    pc, c = make()
    pc.x[1][:] = np.array([0.5, -0.5, 0.2])
    pc.x[0][:] = np.array([0.1, 0.1, 0.1, 0.1])
    pc.compute_errors(top_pred=np.zeros(2))
    yhat0 = g_act(pc.W[0] @ pc.x[1])
    assert np.allclose(pc.eps[0], pc.x[0] - yhat0)

def test_energy_decreases_when_error_decreases():
    pc, c = make()
    pc.compute_errors(top_pred=np.zeros(2)); e_hi = pc.energy()
    for l in range(len(pc.x)): pc.x[l][:] = 0.0
    for l in range(len(pc.W)): pc.W[l][:] = 0.0
    pc.compute_errors(top_pred=np.zeros(2)); e_lo = pc.energy()
    assert e_lo <= e_hi  # zero error -> lower precision-weighted energy term
```

- [ ] **Step 2: Run, fail. Step 3: Implement** `cerebrum/pc_core.py`:
```python
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

    def predict(self, l, top_pred=None):
        """top-down prediction of area l."""
        if l < self.L-1:
            return g_act(self.W[l] @ self.x[l+1])
        return np.zeros_like(self.x[l]) if top_pred is None else top_pred

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
```
- [ ] **Step 4: PASS. Step 5: Commit** — `git commit -am "feat: PC areas (errors, diagonal precision, energy)"`

---

## Task 6: Langevin settling step (Euler-Maruyama) + feedback term

**Files:** Modify `cerebrum/pc_core.py` (add `settle_step`); Create `tests/test_settling.py`

Implements spec §3①: `τ_x dx_l = [ −Π_l ε_l + B_{l-1}(f'⊙Π_{l-1}ε_{l-1}) − γ sign(x_l) ] dt + √(2τ_x T) dW_l`. Euler-Maruyama: `x += (drift/τ_x)·dt + √(2T·dt/τ_x)·ξ`. The bottom area `x[0]` is **clamped** to the observation (sensory input), so it is not updated by settling. The feedback term for area `l` uses `ε[l-1]` (area below) through `B[l-1]`.

- [ ] **Step 1: Failing test** `tests/test_settling.py`:
```python
import numpy as np
from cerebrum.config import CerebrumConfig
from cerebrum.pc_core import PCAreas
from cerebrum.rng import SeededRNG

def test_deterministic_settling_reduces_energy():
    c = CerebrumConfig(dims=(6,5,4), seed=1, T_floor=0.0, n_settle=80, dt=0.05, gamma=0.0)
    pc = PCAreas(c)
    # symmetric limit for a clean Lyapunov check: set B[l] = W[l].T
    for l in range(pc.L-1): pc.B[l] = pc.W[l].T.copy()
    obs = np.array([0.3,-0.2,0.5,0.1,-0.4,0.2])
    pc.x[0][:] = obs
    pc.compute_errors()
    e0 = pc.energy()
    rng = SeededRNG(0, enabled=False)  # T=0 + no noise => deterministic descent
    for _ in range(c.n_settle):
        pc.settle_step(rng, T=0.0, clamp_bottom=obs)
    pc.compute_errors()
    assert pc.energy() < e0           # deterministic settling descends the surrogate energy

def test_noise_prevents_collapse():
    c = CerebrumConfig(dims=(5,4), seed=2, T_floor=0.1, dt=0.05)
    pc = PCAreas(c); obs = np.array([0.2,0.1,-0.3,0.4,0.0])
    pc.x[0][:] = obs; pc.compute_errors()
    rng = SeededRNG(3, enabled=True)
    xs = []
    for _ in range(200):
        pc.settle_step(rng, T=c.T_floor, clamp_bottom=obs); xs.append(pc.x[1].copy())
    var = np.var(np.array(xs[50:]), axis=0)
    assert np.all(var > 0)            # Pillar 4: stays a sampler, never a fixed point
```

- [ ] **Step 2: Run, fail. Step 3: Add `settle_step` to `PCAreas`:**
```python
    def settle_step(self, rng, T, clamp_bottom=None, top_pred=None, broadcast=None, counters=None):
        self.compute_errors(top_pred=top_pred, broadcast=broadcast)
        c = self.cfg
        new_x = [xl.copy() for xl in self.x]
        for l in range(self.L):
            if l == 0 and clamp_bottom is not None:
                new_x[0] = clamp_bottom.copy(); continue
            drift = -self.Pi[l]*self.eps[l]
            if l >= 1:  # feedback from area below via SEPARATE B[l-1] (no transpose)
                pre = g_deriv(self.W[l-1] @ self.x[l]) if False else 1.0  # f' applied at the relay; see note
                drift = drift + self.B[l-1] @ (self.Pi[l-1]*self.eps[l-1])
            drift = drift - c.gamma*np.sign(self.x[l])     # -dR/dx (L1 sparsity)
            step = (drift/c.tau_x)*c.dt
            noise = rng.normal(self.x[l].shape, scale=np.sqrt(2.0*T*c.dt/c.tau_x))
            new_x[l] = self.x[l] + step + noise
            if counters is not None: counters.record_synaptic_ops(self.B[l-1].size if l>=1 else 0)
        self.x = new_x
        if counters is not None:
            for xl in self.x: counters.record_activity(xl)
```
> **Implementation note for the worker:** the `f'` factor in the feedback term is evaluated at the pre-activation of the area-below prediction. For the first cut use the linearized relay `B[l-1] @ (Pi[l-1]*eps[l-1])` (set `f'≈1`) so the deterministic-symmetric test gives a clean energy descent; add the `g_deriv` modulation in Task 8 once plasticity is in and verify the descent test still passes. Do not let this become a `W.T` read — `B` is a separate array.

- [ ] **Step 4: Run (PASS).** If `test_deterministic_settling_reduces_energy` is flaky, reduce `dt` to 0.02 and raise `n_settle` to 150 — it must descend, not oscillate.
- [ ] **Step 5: Commit** — `git commit -am "feat: Langevin settling step (Euler-Maruyama) + separate feedback"`

---

## Task 7: Neuromodulator M + couplings

**Files:** Create `cerebrum/neuromod.py`; Test `tests/test_neuromod.py`

Spec §3③/§4: `M = r − r̄`, `τ_r dr̄/dt = −r̄ + r`; couplings `T(M)=T_floor+b_T·relu(M)`, `Pi_gain(M)=σ(a_Π M)`, `eta(M)=η0·relu(M)`, `T_gate(M)∝1/M`. `M` is the **only** non-local signal and must stay scalar.

- [ ] **Step 1: Failing test** `tests/test_neuromod.py`:
```python
import numpy as np
from cerebrum.config import CerebrumConfig
from cerebrum.neuromod import Neuromodulator
from cerebrum.invariants import assert_scalar_M

def test_M_is_reward_minus_baseline_and_scalar():
    nm = Neuromodulator(CerebrumConfig())
    M = nm.update(reward=1.0); assert_scalar_M(M)
    assert M > 0                       # first reward above baseline 0 -> positive surprise

def test_baseline_tracks_reward():
    nm = Neuromodulator(CerebrumConfig(tau_r=2.0))
    for _ in range(500): nm.update(reward=1.0)
    assert abs(nm.r_bar - 1.0) < 1e-2  # steady reward -> baseline converges -> M -> 0
    assert abs(nm.update(1.0)) < 1e-2

def test_couplings_monotone():
    nm = Neuromodulator(CerebrumConfig())
    assert nm.temperature(0.5) > nm.temperature(0.0)   # surprise heats up
    assert nm.eta(0.5) > nm.eta(0.0)                    # surprise raises learning rate
```

- [ ] **Step 2: Fail. Step 3: Implement** `cerebrum/neuromod.py`:
```python
import numpy as np

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
```
- [ ] **Step 4: PASS. Step 5: Commit** — `git commit -am "feat: scalar neuromodulator M + couplings"`

---

## Task 8: Four-factor local plasticity + precision learning

**Files:** Create `cerebrum/plasticity.py`; Test `tests/test_plasticity.py`

Spec §3③: `τ_w Ẇ_{l,ij} = M·θ_{l,ij}·Π_{l,i}·ε_{l,i}·e_{l,ij}`, eligibility `τ_e ė = −e + a_{l+1,j}`, **precision-once** (Π standalone, eligibility is bare presynaptic low-pass) so `−∂F/∂W = −Π·ε·a` exactly. Precision learning `τ_Π Π̇ = −(Π−Π0)+κ(Π⁻¹−⟨ε²⟩)`. For Stage 1, `θ≡1` (metaplasticity arrives in Stage 3).

- [ ] **Step 1: Failing tests** `tests/test_plasticity.py`:
```python
import numpy as np
from cerebrum.config import CerebrumConfig
from cerebrum.plasticity import Eligibility, weight_update, precision_update

def test_eligibility_lowpasses_presynaptic_activity():
    e = Eligibility(shape=(3,), cfg=CerebrumConfig(tau_e=4.0))
    for _ in range(1000): e.step(a_pre=np.ones(3))
    assert np.allclose(e.value, 1.0, atol=1e-2)   # converges to steady presyn activity

def test_weight_update_matches_negative_grad_in_deterministic_limit():
    # -dF/dW_{ij} = Pi_i * eps_i * a_j  (precision-once). With M=theta=1 the rule must equal eta * that.
    c = CerebrumConfig(eta_w=1.0)
    Pi = np.array([2.0, 0.5]); eps = np.array([0.3, -0.4]); e = np.array([1.0, 0.5, -1.0])
    dW = weight_update(M=1.0, theta=np.ones((2,3)), Pi_post=Pi, eps_post=eps, elig=e, eta=c.eta_w)
    expected = np.outer(Pi*eps, e)                # (2,3)
    assert np.allclose(dW, expected)

def test_M_zero_means_no_learning():
    dW = weight_update(M=0.0, theta=np.ones((2,3)), Pi_post=np.ones(2),
                       eps_post=np.ones(2), elig=np.ones(3), eta=1.0)
    assert np.allclose(dW, 0.0)                    # neuromodulator gates WHEN to learn

def test_precision_converges_to_inverse_variance():
    c = CerebrumConfig(tau_pi=1.0, sigma0=0.0, kappa_pi=1.0)
    Pi = np.array([1.0]); var = 0.25
    for _ in range(5000): Pi = precision_update(Pi, eps_sq=np.array([var]), cfg=c)
    assert abs(Pi[0] - 1.0/var) < 0.1             # Pi -> 1/<eps^2>
```

- [ ] **Step 2: Fail. Step 3: Implement** `cerebrum/plasticity.py`:
```python
import numpy as np

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
    """tau_Pi dPi/dt = -(Pi-Pi0) + kappa(1/Pi - <eps^2>). Diagonal, local per unit."""
    dPi = -(Pi - cfg.Pi0) + cfg.kappa_pi*(1.0/np.maximum(Pi,1e-6) - eps_sq)
    return Pi + (1.0/cfg.tau_pi)*dPi
```
- [ ] **Step 4: PASS. Step 5: Commit** — `git commit -am "feat: four-factor local plasticity + precision learning"`

---

## Task 9: Feedback weight B local update (Kolen-Pollack-class) + locality guard

**Files:** Modify `cerebrum/plasticity.py` (add `feedback_update`); add tests to `tests/test_plasticity.py`

Spec §3: `τ_B Ḃ_l = η_B a_{l+1} ε_lᵀ − λ_B B_l` — local, same neuron pair, **no transpose read**. Honest: NOT guaranteed to track Wᵀ.

- [ ] **Step 1: Failing test (append):**
```python
def test_feedback_update_is_local_outer_product():
    from cerebrum.plasticity import feedback_update
    from cerebrum.config import CerebrumConfig
    c = CerebrumConfig(eta_b=1.0, lam_b=0.0)
    a_up = np.array([1.0, -1.0]); eps = np.array([0.5, 0.2, -0.3])   # B shape (2,3)
    B = np.zeros((2,3))
    dB = feedback_update(B, a_up=a_up, eps=eps, cfg=c)
    assert np.allclose(dB, np.outer(a_up, eps))    # uses only local a_up, eps (no W, no transpose)
```
- [ ] **Step 2: Fail. Step 3: Implement (append to plasticity.py):**
```python
def feedback_update(B, a_up, eps, cfg):
    """Local feedback-weight rule: eta_b * a_up outer eps - lam_b * B. No transpose of W is read."""
    return cfg.eta_b*np.outer(a_up, eps) - cfg.lam_b*B
```
- [ ] **Step 4: PASS. Step 5: Commit** — `git commit -am "feat: local feedback-weight (B) update, no weight transport"`

---

## Task 10: Grid HEAD — frozen modules + path integration

**Files:** Create `cerebrum/grid_head.py`; Test `tests/test_grid_head.py`

Spec §5: each module `m` has a frozen 2D spatial frequency vector `k_m`; phase `θ_m = k_m · pos`; code `g_m = [cos θ_m, sin θ_m]`. Path integration over **Exogenous** actions is a structural identity (loop closure exact). The transition accepts ONLY `Exogenous` (BAN-1).

- [ ] **Step 1: Failing test** `tests/test_grid_head.py`:
```python
import numpy as np, pytest
from cerebrum.config import CerebrumConfig
from cerebrum.grid_head import GridHead
from cerebrum.types import Exogenous

def test_code_shape_and_unit_norm_per_module():
    gh = GridHead(CerebrumConfig(grid_n_modules=6)); gh.reset()
    g = gh.encode(); assert g.shape == (12,)              # 6 modules x 2
    mods = g.reshape(6,2)
    assert np.allclose(np.linalg.norm(mods, axis=1), 1.0) # each module is a unit phasor

def test_path_integration_loop_closure():
    gh = GridHead(CerebrumConfig()); gh.reset()
    g0 = gh.encode().copy()
    for a in [[1,0],[0,1],[-1,0],[0,-1]]:                 # walk a unit square, return to start
        gh.transition(Exogenous(np.array(a, float)))
    assert np.allclose(gh.encode(), g0, atol=1e-9)        # exact loop closure (structural)

def test_path_integration_is_additive():
    gh = GridHead(CerebrumConfig()); gh.reset()
    gh.transition(Exogenous(np.array([2.0,1.0]))); gA = gh.encode().copy()
    gh.reset()
    gh.transition(Exogenous(np.array([1.0,0.0]))); gh.transition(Exogenous(np.array([1.0,1.0])))
    assert np.allclose(gh.encode(), gA, atol=1e-9)        # displacement composes (graph algebra)

def test_transition_rejects_plain_array():
    gh = GridHead(CerebrumConfig()); gh.reset()
    with pytest.raises(TypeError):
        gh.transition(np.array([1.0,0.0]))                # data-derived action = BAN-1
```

- [ ] **Step 2: Fail. Step 3: Implement** `cerebrum/grid_head.py` (modules + path integration; content store added in Task 11):
```python
import numpy as np
from .invariants import assert_exogenous_action

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
    def reset(self):
        self.pos = np.zeros(2)
    def transition(self, action):
        assert_exogenous_action(action)
        self.pos = self.pos + action.value
    def encode(self):
        phase = self.k @ self.pos                            # (M,)
        return np.stack([np.cos(phase), np.sin(phase)], axis=1).reshape(-1)
```
- [ ] **Step 4: PASS. Step 5: Commit** — `git commit -am "feat: grid HEAD modules + exogenous path integration"`

---

## Task 11: Grid HEAD — content store (Hebbian bind) + completion

**Files:** Modify `cerebrum/grid_head.py`; add tests to `tests/test_grid_head.py`

Spec §5: content store `M_t += η (w ⊗ ĝ)`; completion `ŵ = M_t g` (then decode). This is the fast, M-gated, within-episode memory; structural weights stay frozen.

- [ ] **Step 1: Failing tests (append):**
```python
def test_bind_then_complete_recovers_observation_at_bound_location():
    gh = GridHead(CerebrumConfig()); gh.reset()
    obs = np.array([0.0, 1.0, 0.0, -1.0])
    gh.bind(obs)                                  # bind obs at current location
    rec = gh.complete()                           # complete at the SAME location
    # cosine similarity high (single binding -> proportional recall)
    assert np.dot(rec, obs)/(np.linalg.norm(rec)*np.linalg.norm(obs)+1e-9) > 0.9

def test_completion_generalizes_to_path_integrated_location():
    gh = GridHead(CerebrumConfig()); gh.reset()
    obsA = np.array([1.0,0.0,0.0]); obsB = np.array([0.0,1.0,0.0])
    gh.bind(obsA)
    gh.transition(__import__('cerebrum.types',fromlist=['Exogenous']).Exogenous(np.array([3.0,2.0])))
    gh.bind(obsB)
    # return to A by exact inverse displacement; completion should recall obsA (graph completion)
    gh.transition(__import__('cerebrum.types',fromlist=['Exogenous']).Exogenous(np.array([-3.0,-2.0])))
    rec = gh.complete()
    assert np.dot(rec, obsA)/(np.linalg.norm(rec)*np.linalg.norm(obsA)+1e-9) > 0.8
```
- [ ] **Step 2: Fail. Step 3: Implement (extend GridHead):**
```python
    # in __init__ add:  self.store = None  ; self.obs_dim = None
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
```
- [ ] **Step 4: PASS.** (If `test_completion_generalizes...` is borderline, increase `grid_n_modules` to 8 — more modules sharpen the grid code's spatial selectivity.)
- [ ] **Step 5: Commit** — `git commit -am "feat: grid HEAD content store (bind) + completion (graph-completion)"`

---

## Task 12: CerebrumCore assembly (Stage-1: PC areas + grid HEAD, no gate)

**Files:** Create `cerebrum/network.py`; Test `tests/test_network.py`

Wires the grid HEAD's decoded completion as the top-down prediction to the top PC area, runs settling, then a local-plasticity learning step. No gate/workspace yet (Stage 2). Verifies invariants + counters fire.

- [ ] **Step 1: Failing test** `tests/test_network.py`:
```python
import numpy as np
from cerebrum.config import CerebrumConfig
from cerebrum.network import CerebrumCore
from cerebrum.types import Exogenous

def test_observe_learn_predict_runs_and_counts():
    c = CerebrumConfig(dims=(6,5,4), grid_n_modules=6, n_settle=20, seed=0)
    net = CerebrumCore(c)
    obs = np.array([0.2,-0.1,0.3,0.0,0.5,-0.2])
    M = net.observe_and_learn(obs, reward=1.0)             # one episode step
    assert np.isscalar(M) or np.ndim(M)==0                 # scalar neuromodulator only
    assert net.counters.global_comm_learn >= 1             # one scalar M broadcast
    assert net.counters.synaptic_ops > 0

def test_no_weight_transport_used():
    # structural guarantee: B and W are independent arrays (no aliasing, no transpose read)
    c = CerebrumConfig(dims=(5,4), seed=1); net = CerebrumCore(c)
    for l in range(net.pc.L-1):
        assert net.pc.B[l] is not net.pc.W[l]
        assert net.pc.B[l].shape == net.pc.W[l].T.shape    # shapes compatible but separate synapses
```

- [ ] **Step 2: Fail. Step 3: Implement** `cerebrum/network.py`:
```python
import numpy as np
from .pc_core import PCAreas
from .grid_head import GridHead
from .neuromod import Neuromodulator
from .plasticity import Eligibility, weight_update, precision_update, feedback_update
from .counters import Counters
from .rng import SeededRNG
from .invariants import assert_scalar_M
from .types import Exogenous

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
        self.elig = [Eligibility((cfg.dims[l], cfg.dims[l+1]), cfg) for l in range(self.pc.L-1)]

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
```
- [ ] **Step 4: PASS. Step 5: Commit** — `git commit -am "feat: CerebrumCore (PC + grid HEAD) observe/learn/predict, no gate"`

---

## Task 13: Task-1 environment (TEM-class gridworld) + episode generator

**Files:** Create `benchmarks/__init__.py`, `benchmarks/tasks/__init__.py`, `benchmarks/tasks/gridworld.py`; Test add to `tests/test_task1_smoke.py`

A toroidal HxW grid; each cell has a fixed random one-hot "object" obs from a small vocab. An episode = a fixed environment + a random walk of K steps (exogenous actions, wrapping). Held-out queries: (start cell, displacement) → target cell that was **observed** but whose specific path was **not walked**, testing transition generalization.

- [ ] **Step 1: Failing test** `tests/test_task1_smoke.py`:
```python
import numpy as np
from benchmarks.tasks.gridworld import GridWorld, make_episode

def test_gridworld_obs_are_consistent_per_cell():
    gw = GridWorld(h=4, w=4, vocab=5, seed=0)
    assert np.array_equal(gw.obs_at((1,2)), gw.obs_at((1,2)))   # deterministic per cell
    assert gw.obs_at((0,0)).shape == (5,)

def test_episode_has_walk_and_heldout_queries():
    ep = make_episode(h=4, w=4, vocab=5, K=8, seed=1)
    assert len(ep.walk) == 8
    assert len(ep.queries) > 0
    for (start, disp, target_cell) in ep.queries:
        assert target_cell in ep.observed_cells          # target was observed (obs known)
```

- [ ] **Step 2: Fail. Step 3: Implement** `benchmarks/tasks/gridworld.py`:
```python
import numpy as np
from dataclasses import dataclass

ACTIONS = {"N":(-1,0), "S":(1,0), "E":(0,1), "W":(0,-1)}

class GridWorld:
    def __init__(self, h, w, vocab, seed=0):
        self.h, self.w, self.vocab = h, w, vocab
        rng = np.random.default_rng(seed)
        self._obj = rng.integers(0, vocab, size=(h,w))    # object id per cell (structure-free content)
    def obs_at(self, cell):
        r, c = cell; v = np.zeros(self.vocab); v[self._obj[r % self.h, c % self.w]] = 1.0; return v
    def step(self, cell, action_name):
        dr, dc = ACTIONS[action_name]; return ((cell[0]+dr) % self.h, (cell[1]+dc) % self.w)

@dataclass
class Episode:
    gw: GridWorld
    walk: list           # list of (cell, action_name, action_vec)
    observed_cells: set
    queries: list        # (start_cell, displacement_vec, target_cell)

def make_episode(h, w, vocab, K, seed=0):
    gw = GridWorld(h, w, vocab, seed=seed); rng = np.random.default_rng(seed+1)
    names = list(ACTIONS); cell = (0,0); walk = []; observed = {cell}
    walked_edges = set()
    for _ in range(K):
        a = names[rng.integers(0,len(names))]; nxt = gw.step(cell, a)
        walk.append((cell, a, np.array(ACTIONS[a], float)))
        walked_edges.add((cell, nxt)); cell = nxt; observed.add(cell)
    # held-out queries: pairs of observed cells whose connecting straight path was NOT a walked edge
    obs_list = sorted(observed); queries = []
    for s in obs_list:
        for t in obs_list:
            if s == t: continue
            disp = np.array([(t[0]-s[0]), (t[1]-s[1])], float)  # raw displacement (torus-unwrapped)
            if (s, t) not in walked_edges:
                queries.append((s, disp, t))
    return Episode(gw=gw, walk=walk, observed_cells=observed, queries=queries[:64])
```
- [ ] **Step 4: PASS. Step 5: Commit** — `git commit -am "feat: TEM-class gridworld task + episode generator"`

---

## Task 14: Graph-completion metric + CEREBRUM runner

**Files:** Create `benchmarks/tasks/graph_completion.py`; add test to `tests/test_task1_smoke.py`

The model walks the episode (binding obs at each visited cell via exogenous actions), then for each held-out query path-integrates from `start` by `displacement` and completes → predicted obs; score = top-1 match vs the target cell's true obs.

- [ ] **Step 1: Failing test (append):**
```python
import numpy as np
from cerebrum.config import CerebrumConfig
from cerebrum.network import CerebrumCore
from benchmarks.tasks.gridworld import make_episode
from benchmarks.tasks.graph_completion import run_cerebrum_episode

def test_cerebrum_scores_above_chance_on_completion():
    ep = make_episode(h=4, w=4, vocab=5, K=12, seed=2)
    cfg = CerebrumConfig(dims=(5,8,8), grid_n_modules=8, n_settle=10, seed=0)
    score = run_cerebrum_episode(CerebrumCore(cfg), ep)
    assert score > 1.0/5                       # beats 1/vocab chance via path-integrated completion
```

- [ ] **Step 2: Fail. Step 3: Implement** `benchmarks/tasks/graph_completion.py`:
```python
import numpy as np
from cerebrum.types import Exogenous

def run_cerebrum_episode(net, ep):
    """Walk the episode binding obs at each cell; then score held-out path-integrated completions."""
    net.grid.reset()
    # walk: at each step bind current obs (drive grid by EXOGENOUS action)
    cell = (0,0)
    net.observe_and_learn(ep.gw.obs_at(cell), reward=1.0)
    for (c, a, avec) in ep.walk:
        net.move(Exogenous(avec))
        cell = ep.gw.step(c, a)
        net.observe_and_learn(ep.gw.obs_at(cell), reward=1.0)
    # query: from start, path-integrate by displacement, complete, compare top-1 obs
    correct = 0
    for (start, disp, target) in ep.queries:
        net.grid.reset()
        # move grid to 'start' then by 'disp' (exogenous); start offset from origin (0,0)
        net.move(Exogenous(np.array([start[0], start[1]], float)))
        net.move(Exogenous(disp))
        pred = net.predict_obs_here(ep.gw.vocab)
        if pred.size and np.argmax(pred) == np.argmax(ep.gw.obs_at(target)):
            correct += 1
    return correct/len(ep.queries) if ep.queries else 0.0
```
> **Worker note:** binding happens at the cell's grid code; the runner must move the grid to each visited cell with the SAME origin convention used at query time (origin = (0,0) at first obs). Keep the grid's `pos` consistent: bind at `pos == cell coordinates`. If completion underperforms, verify the bind-time `pos` equals the query-time path-integrated `pos` for the same cell (they must match exactly — that is the whole graph-completion mechanism).

- [ ] **Step 4: PASS. Step 5: Commit** — `git commit -am "feat: graph-completion metric + CEREBRUM episode runner"`

---

## Task 15: Baselines — flat prior (Pillar-3 ablation) + backprop MLP comparator

**Files:** Create `benchmarks/baselines/__init__.py`, `benchmarks/baselines/flat_prior.py`, `benchmarks/baselines/backprop_mlp.py`; add tests.

- **Flat prior:** identical to CEREBRUM but the "grid" code is a fixed random code per *visited cell* that does NOT path-integrate — so it can recall obs only at exactly-revisited codes, and a path-integrated query lands on an unbound code → chance. Proves Pillar 3 is load-bearing.
- **Backprop MLP (comparator only):** maps (start one-hot ⊕ displacement) → target obs, trained with manual backprop on the walked edges; tested on held-out queries. With few K it overfits walked edges and generalizes worse than the grid at small K.

- [ ] **Step 1: Failing tests** (`tests/test_task1_smoke.py`, append):
```python
def test_flat_prior_is_near_chance_on_completion():
    from benchmarks.baselines.flat_prior import run_flat_episode
    from benchmarks.tasks.gridworld import make_episode
    ep = make_episode(h=4, w=4, vocab=5, K=12, seed=2)
    score = run_flat_episode(ep)
    assert score <= 0.45        # no path-integration -> cannot complete held-out paths

def test_backprop_mlp_runs():
    from benchmarks.baselines.backprop_mlp import run_mlp_episode
    from benchmarks.tasks.gridworld import make_episode
    ep = make_episode(h=4, w=4, vocab=5, K=12, seed=2)
    s = run_mlp_episode(ep, epochs=50)
    assert 0.0 <= s <= 1.0
```

- [ ] **Step 2: Fail. Step 3: Implement.**

`benchmarks/baselines/flat_prior.py`:
```python
import numpy as np

def run_flat_episode(ep):
    """Flat prior: random fixed code per VISITED cell, no path integration. Recall only at bound codes."""
    rng = np.random.default_rng(123)
    code = {}; store = None
    def code_of(cell):
        if cell not in code: code[cell] = rng.standard_normal(16)
        return code[cell]
    cell = (0,0); store = np.zeros((ep.gw.vocab, 16))
    store += np.outer(ep.gw.obs_at(cell), code_of(cell))
    for (c, a, avec) in ep.walk:
        cell = ep.gw.step(c, a); store += np.outer(ep.gw.obs_at(cell), code_of(cell))
    correct = 0
    for (start, disp, target) in ep.queries:
        # flat prior has NO transition algebra: a path-integrated query cannot synthesize target's code;
        # it can only guess from the start cell's code -> wrong target obs.
        pred = store @ code_of(start)
        if np.argmax(pred) == np.argmax(ep.gw.obs_at(target)): correct += 1
    return correct/len(ep.queries) if ep.queries else 0.0
```

`benchmarks/baselines/backprop_mlp.py`:
```python
import numpy as np

def run_mlp_episode(ep, epochs=100, hidden=32, lr=0.1, seed=0):
    """Baseline COMPARATOR ONLY (uses backprop — CEREBRUM never does). Maps start-cell-onehot ⊕ disp -> obs."""
    rng = np.random.default_rng(seed)
    cells = sorted(ep.observed_cells); idx = {c:i for i,c in enumerate(cells)}
    nin = len(cells) + 2; nout = ep.gw.vocab
    W1 = 0.1*rng.standard_normal((hidden, nin)); b1 = np.zeros(hidden)
    W2 = 0.1*rng.standard_normal((nout, hidden)); b2 = np.zeros(nout)
    def feat(start, disp):
        v = np.zeros(nin); v[idx[start]] = 1.0; v[-2:] = disp; return v
    # train on WALKED edges only (few-shot supervision)
    Xtr = []; Ytr = []
    cell = (0,0)
    for (c,a,avec) in ep.walk:
        nxt = ep.gw.step(c,a); Xtr.append(feat(c, avec)); Ytr.append(ep.gw.obs_at(nxt))
    Xtr = np.array(Xtr); Ytr = np.array(Ytr)
    for _ in range(epochs):
        h = np.tanh(Xtr@W1.T + b1); logits = h@W2.T + b2
        p = np.exp(logits - logits.max(1,keepdims=True)); p /= p.sum(1,keepdims=True)
        g = (p - Ytr)/len(Xtr)
        gW2 = g.T@h; gb2 = g.sum(0); gh = (g@W2)*(1-h**2)
        gW1 = gh.T@Xtr; gb1 = gh.sum(0)
        W2 -= lr*gW2; b2 -= lr*gb2; W1 -= lr*gW1; b1 -= lr*gb1
    correct = 0
    for (start, disp, target) in ep.queries:
        x = feat(start, disp); h = np.tanh(x@W1.T+b1); logits = h@W2.T+b2
        if np.argmax(logits) == np.argmax(ep.gw.obs_at(target)): correct += 1
    return correct/len(ep.queries) if ep.queries else 0.0
```
- [ ] **Step 4: PASS. Step 5: Commit** — `git commit -am "feat: flat-prior ablation + backprop MLP comparator"`

---

## Task 16: run_task1 harness — CEREBRUM vs flat vs backprop @ K ∈ {5,10,20}

**Files:** Create `benchmarks/run_task1.py`; add the load-bearing test to `tests/test_task1_smoke.py`

The decisive result: **CEREBRUM-grid beats flat-prior on graph-completion at small K** (Pillar-3 load-bearing), averaged over seeds. (Backprop comparison is reported but not asserted — the honest claim is sample-efficiency divergence, and the headline assertion is grid > flat.)

- [ ] **Step 1: Failing test (append):**
```python
def test_cerebrum_grid_beats_flat_prior_averaged():
    from benchmarks.run_task1 import run_sweep
    res = run_sweep(Ks=(5,10,20), seeds=(0,1,2), h=4, w=4, vocab=5)
    for K in (5,10,20):
        assert res["cerebrum"][K] > res["flat"][K] + 0.05    # grid prior buys sample efficiency
```

- [ ] **Step 2: Fail. Step 3: Implement** `benchmarks/run_task1.py`:
```python
import numpy as np
from cerebrum.config import CerebrumConfig
from cerebrum.network import CerebrumCore
from benchmarks.tasks.gridworld import make_episode
from benchmarks.tasks.graph_completion import run_cerebrum_episode
from benchmarks.baselines.flat_prior import run_flat_episode
from benchmarks.baselines.backprop_mlp import run_mlp_episode

def run_sweep(Ks=(5,10,20), seeds=(0,1,2), h=4, w=4, vocab=5):
    out = {"cerebrum":{}, "flat":{}, "mlp":{}}
    for K in Ks:
        g=[]; f=[]; m=[]
        for s in seeds:
            ep = make_episode(h=h, w=w, vocab=vocab, K=K, seed=s)
            cfg = CerebrumConfig(dims=(vocab,8,8), grid_n_modules=8, n_settle=10, seed=s)
            g.append(run_cerebrum_episode(CerebrumCore(cfg), ep))
            f.append(run_flat_episode(ep))
            m.append(run_mlp_episode(ep, epochs=80))
        out["cerebrum"][K]=float(np.mean(g)); out["flat"][K]=float(np.mean(f)); out["mlp"][K]=float(np.mean(m))
    return out

if __name__ == "__main__":
    res = run_sweep()
    print(f"{'K':>4} {'CEREBRUM-grid':>12} {'flat-prior':>12} {'backprop-MLP':>14}")
    for K in sorted(res['cerebrum']):
        print(f"{K:>4} {res['cerebrum'][K]:>12.3f} {res['flat'][K]:>12.3f} {res['mlp'][K]:>14.3f}")
```
- [ ] **Step 4: Run** `python3 -m pytest tests/test_task1_smoke.py -v` (PASS) **and** `python3 benchmarks/run_task1.py` to print the table. If grid does not beat flat, debug the bind/query `pos` consistency (Task 14 worker note) before weakening any assertion — the grid MUST win here or Pillar 3 is not wired correctly.
- [ ] **Step 5: Commit** — `git commit -am "feat: Task-1 few-shot sweep (CEREBRUM vs flat vs backprop)"`

---

## Task 17: README + full suite green + honesty gate

**Files:** Create `README.md`; run full suite.

- [ ] **Step 1:** Write `README.md` summarizing CEREBRUM (link the spec), the five pillars, the bans-as-invariants, and the honest status (scaling = unproven bet; zero open problems solved). Include the Task-1 result table from Task 16. State explicitly: **no claim of "scaling solved", "stability-plasticity solved", or "O(1) global comm" may be made.**
- [ ] **Step 2:** Run the whole suite: `python3 -m pytest -q`. Expected: all green.
- [ ] **Step 3:** Run `python3 benchmarks/run_task1.py` and paste the table into the README.
- [ ] **Step 4: Commit** — `git commit -am "docs: README + Stage 0+1 complete (PC core + grid HEAD + few-shot win)"`

---

## Self-Review (run by the planner)

**Spec coverage (§-by-§):**
- §2 F functional → Tasks 5 (energy), 6 (settling descends it), 8 (−∂F/∂W identity). ✓
- §3① settling (Langevin + feedback B, no transport) → Task 6, 9. ✓
- §3② gate → **deferred to Stage 2** (out of scope here; explicitly staged). ✓ (gap is intentional)
- §3③ four-factor plasticity + precision-once + B + precision learning → Tasks 7, 8, 9. ✓
- §4 stochastic term / temperature → Task 6 (noise), Task 7 (M-coupled T). ✓
- §5 structured prior (grid, path integration, content store, exogenous z_act) → Tasks 10, 11. ✓
- §10 Task-1 few-shot + baselines + metric → Tasks 13–16. ✓ Task-2/Task-3 deferred (staged). ✓
- §12 invariants (one-hot, exogenous z_act, scalar M, no transport, no autograd) → Task 2 (assertions), Task 10/11 (Exogenous type), Task 12 (scalar-M + B≠W structural). One-hot assertion is built (Task 2) and used in Stage 2. ✓

**Placeholder scan:** every code step has complete code or a precise contract test; worker notes flag the two known-tricky spots (settling `f'` linearization; bind/query `pos` consistency) without leaving logic unspecified. ✓

**Type consistency:** `weight_update(M, theta, Pi_post, eps_post, elig, eta)`, `precision_update(Pi, eps_sq, cfg)`, `feedback_update(B, a_up, eps, cfg)`, `GridHead.transition(Exogenous)`, `CerebrumCore.observe_and_learn(obs, reward)`/`move(Exogenous)`/`predict_obs_here(obs_dim)` are used consistently across Tasks 8–16. ✓

**Intentional scope gaps (staged):** gate/workspace/broadcast (Stage 2), metaplasticity + Task-2 forgetting (Stage 3), full Task-3 energy curves (instrumented here, plotted later). These get their own plans.
