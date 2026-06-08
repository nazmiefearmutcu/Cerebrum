# GRAIL Stage 3 Implementation Plan — Surprise-Gated Metaplastic Fuse + Catastrophic-Forgetting (Task-2)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes.

**Goal:** Build the surprise-gated per-synapse metaplastic fuse `θ/c` (the OP3 "addressed-not-solved" mechanism) and prove on a sequential reconstruction stream A→B→C (no replay, no iid batch, no task-boundary signal) that it mitigates catastrophic forgetting better than always-plastic local learning and competitively with an EWC analog — WITHOUT a Fisher pass or stored anchor weights, reusing the SAME `ε` already computed for inference.

**Architecture:** Builds on Stage 0+1 local plasticity (`grail/plasticity.py`'s `weight_update` already accepts a `theta` gate; Stage 1 used `θ≡1`). The new `MetaplasticFuse` maintains a per-synapse consolidation reserve `c` and surprise baseline `S̄`, and emits the plasticity-permission `θ`. A small continual-learning harness (PC reconstruction) drives the demonstration. Pure NumPy, no autograd.

**Tech Stack:** Python 3, numpy, pytest. No torch/jax/sklearn in `grail/`.

**Spec:** `docs/superpowers/specs/2026-06-08-grail-cortical-workspace-design.md` §3③ (metaplastic fuse), §7 OP3, §9 FM4, §10 Task-2.

**Honesty gate (enforced in README):** OP3 is "genuinely addressed, pending this stage's validation — NOT solved." The fuse has no stability proof and two named failure modes (forgetting OR plastic-death). Do NOT write "stability-plasticity solved."

---

## File Structure (additions)

```
grail/
  config.py        # MODIFY: add metaplasticity fields (tau_S, tau_c, alpha_c, beta_c, c_max, g_theta)
  metaplasticity.py# MetaplasticFuse: per-synapse c, S̄, theta(surprise) — reuses Pi,eps,eligibility
tests/
  test_metaplasticity.py  test_stage3_smoke.py
benchmarks/
  tasks/continual.py        # sequential A->B->C reconstruction stream + forgetting metric + GRAIL runner
  baselines/ewc.py          # EWC-analog (quadratic anchor) baseline on the same local substrate
  run_stage3.py             # forgetting comparison: GRAIL-fuse vs theta=1 vs EWC-analog
```

---

## Task 1: Config — metaplasticity hyperparameters

**Files:** Modify `grail/config.py`; Test `tests/test_metaplasticity.py` (first test stub)

- [ ] **Step 1: Failing test** `tests/test_metaplasticity.py`:
```python
from grail.config import GRAILConfig

def test_metaplasticity_config_present():
    c = GRAILConfig()
    assert c.c_max > 0
    assert c.tau_c > c.tau_S      # consolidation slower than the surprise baseline EMA
    assert c.alpha_c > 0 and c.beta_c > 0 and c.g_theta > 0
```

- [ ] **Step 2: Run, fail. Step 3: Implement** — add fields to the `GRAILConfig` dataclass (keep existing fields unchanged):
```python
    # metaplasticity (Stage 3)
    tau_S: float = 20.0       # surprise-baseline EMA timescale
    tau_c: float = 300.0      # consolidation-reserve timescale (slow)
    alpha_c: float = 1.0      # low-surprise consolidation gain (builds c)
    beta_c: float = 1.5       # high-surprise erosion gain (frees c)
    c_max: float = 1.0        # max consolidation reserve
    g_theta: float = 4.0      # plasticity-permission sigmoid sharpness
```
- [ ] **Step 4: PASS. Step 5: Commit** — `git commit -am "feat(stage3): metaplasticity config fields"`

---

## Task 2: MetaplasticFuse — surprise, consolidation reserve, plasticity permission

**Files:** Create `grail/metaplasticity.py`; add tests to `tests/test_metaplasticity.py`

Spec §3③/§7: per-synapse surprise `S_ij = Π_i|ε_i·e_j| − S̄_ij`; reserve `τ_c ċ = +α[S]_-(c_max−c) − β[S]_+ c` (`[S]_-` = below-baseline/predictive builds c; `[S]_+` = above-baseline/surprising erodes c); permission `θ = σ(g(S − c))` (surprise opens the fuse, consolidation closes it). The surprise REUSES the same `Π,ε,eligibility` from inference — no Fisher pass, no task-boundary signal, no stored anchor weights.

- [ ] **Step 1: Failing tests (append):**
```python
import numpy as np
from grail.config import GRAILConfig
from grail.metaplasticity import MetaplasticFuse

def test_sustained_low_surprise_builds_reserve_and_closes_fuse():
    c = GRAILConfig(tau_c=5.0, tau_S=2.0)
    fuse = MetaplasticFuse(shape=(2,2), cfg=c)
    Pi = np.array([1.0,1.0]); eps = np.array([0.0,0.0]); elig = np.array([0.0,0.0])  # perfectly predicted -> low surprise
    th = None
    for _ in range(500): th = fuse.update(Pi, eps, elig)
    assert np.all(fuse.c > 0.5)            # reserve builds under sustained low surprise
    assert np.all(th < 0.5)                # fuse closes (consolidated synapses freeze)

def test_high_surprise_opens_fuse():
    c = GRAILConfig(tau_c=5.0, tau_S=2.0)
    fuse = MetaplasticFuse(shape=(2,2), cfg=c)
    # first consolidate under low surprise
    for _ in range(500): fuse.update(np.array([1.,1.]), np.array([0.,0.]), np.array([0.,0.]))
    # then a surprising event (large eps*elig) should push theta back up
    th = fuse.update(np.array([1.,1.]), np.array([3.0,3.0]), np.array([3.0,3.0]))
    assert np.all(th > 0.3)                # surprise reopens plasticity (learn-on-surprise)

def test_theta_in_unit_interval():
    fuse = MetaplasticFuse(shape=(3,4), cfg=GRAILConfig())
    th = fuse.update(np.ones(3), np.random.default_rng(0).standard_normal(3), np.ones(4))
    assert th.shape == (3,4) and np.all(th >= 0) and np.all(th <= 1)
```

- [ ] **Step 2: Fail. Step 3: Implement** `grail/metaplasticity.py`:
```python
import numpy as np

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
        S = S_raw - self.S_bar                                       # surprise relative to baseline
        self.S_bar += (1.0/self.cfg.tau_S) * (S_raw - self.S_bar)    # baseline EMA
        pos = np.maximum(S, 0.0)                                     # [S]_+ erodes c
        neg = np.maximum(-S, 0.0)                                    # [S]_- builds c
        dc = self.cfg.alpha_c*neg*(self.cfg.c_max - self.c) - self.cfg.beta_c*pos*self.c
        self.c = np.clip(self.c + (1.0/self.cfg.tau_c)*dc, 0.0, self.cfg.c_max)
        theta = 1.0/(1.0 + np.exp(-self.cfg.g_theta*(S - self.c)))   # sigma(g(S - c))
        return theta
```
> **Worker note:** `test_high_surprise_opens_fuse` checks that after consolidation a large `eps*elig` raises θ. If θ does not rise enough, it is because `S = S_raw - S̄` and the baseline has risen; the single surprising step still gives `S_raw ≫ S̄` so `S` is large positive → θ→1. Do not weaken the assertion; if needed raise `g_theta` slightly (sigmoid sharpness) — it is a config knob, not an assertion change.

- [ ] **Step 4: PASS. Step 5: Commit** — `git commit -am "feat(stage3): surprise-gated metaplastic fuse (c reserve + theta permission)"`

---

## Task 3: Continual-learning harness — sequential reconstruction A→B→C + forgetting metric

**Files:** Create `benchmarks/__init__.py` (if missing), `benchmarks/tasks/continual.py`; Test `tests/test_stage3_smoke.py`

A PC reconstruction network (`PCAreas`, 2 areas) learns to predict structured inputs online. Three tasks = three disjoint clusters of prototype patterns (`A`,`B`,`C`); the model streams A's patterns (settle + local-plastic update, optionally fuse-gated), then B, then C — NO replay, NO iid mixing, NO task-boundary signal to the model. **Metric:** reconstruction error (settled `‖ε₀‖²`) on held-out A patterns measured after A, after B, after C. **Forgetting** = (error-on-A after C) − (error-on-A after A).

- [ ] **Step 1: Failing test** `tests/test_stage3_smoke.py`:
```python
import numpy as np
from benchmarks.tasks.continual import run_continual

def test_fuse_reduces_forgetting_vs_always_plastic():
    fused   = run_continual(use_fuse=True,  seed=0)
    plastic = run_continual(use_fuse=False, seed=0)
    # both learn A well initially
    assert fused["errA_afterA"] < 1.0 and plastic["errA_afterA"] < 1.0
    # the metaplastic fuse forgets A LESS than always-plastic local learning
    assert fused["forgetA"] < plastic["forgetA"]
    # ...while still learning C (not frozen solid / plastic-death)
    assert fused["errC_afterC"] < fused["errC_beforeC"]
```

- [ ] **Step 2: Fail. Step 3: Implement** `benchmarks/tasks/continual.py`:
```python
import numpy as np
from dataclasses import replace
from grail.config import GRAILConfig
from grail.pc_core import PCAreas
from grail.plasticity import Eligibility, weight_update, precision_update, feedback_update
from grail.metaplasticity import MetaplasticFuse
from grail.neuromod import Neuromodulator
from grail.rng import SeededRNG

def _prototypes(rng, n_tasks, per_task, dim):
    return [ [rng.standard_normal(dim) for _ in range(per_task)] for _ in range(n_tasks) ]

def _err_on(net, patterns, cfg, rng):
    tot = 0.0
    for p in patterns:
        for _ in range(cfg.n_settle): net.settle_step(rng, T=cfg.T_floor, clamp_bottom=p)
        net.compute_errors(); tot += float(np.sum(net.eps[0]**2))
    return tot/len(patterns)

def run_continual(use_fuse, seed=0, dim=10, per_task=6, passes=40):
    cfg = GRAILConfig(dims=(dim, 8), n_settle=10, seed=seed, tau_w=40.0)
    rng_proto = np.random.default_rng(seed+5)
    A,B,C = _prototypes(rng_proto, 3, per_task, dim)
    net = PCAreas(cfg); nm = Neuromodulator(cfg); rng = SeededRNG(seed)
    elig = [Eligibility((cfg.dims[l+1],), cfg) for l in range(net.L-1)]
    fuse = [MetaplasticFuse(net.W[l].shape, cfg) for l in range(net.L-1)] if use_fuse else None

    def train(patterns):
        for _ in range(passes):
            for p in patterns:
                for _ in range(cfg.n_settle): net.settle_step(rng, T=cfg.T_floor, clamp_bottom=p)
                net.compute_errors()
                M = nm.update(reward=1.0)
                for l in range(net.L-1):
                    elig[l].step(a_pre=net.x[l+1])
                    theta = fuse[l].update(net.Pi[l], net.eps[l], elig[l].value) if use_fuse else np.ones_like(net.W[l])
                    net.W[l] += weight_update(M=M, theta=theta, Pi_post=net.Pi[l], eps_post=net.eps[l],
                                              elig=elig[l].value, eta=cfg.eta_w/cfg.tau_w)
                    net.B[l] += (1.0/cfg.tau_b)*feedback_update(net.B[l], a_up=net.x[l+1], eps=net.eps[l], cfg=cfg)
                    net.Pi[l] = precision_update(net.Pi[l], eps_sq=net.eps[l]**2, cfg=cfg)

    train(A); errA_afterA = _err_on(net, A, cfg, rng)
    errC_beforeC = _err_on(net, C, cfg, rng)
    train(B); train(C)
    errA_afterC = _err_on(net, A, cfg, rng); errC_afterC = _err_on(net, C, cfg, rng)
    cbar = float(np.mean([f.c.mean() for f in fuse])) if use_fuse else 0.0
    return {"errA_afterA": errA_afterA, "errA_afterC": errA_afterC,
            "forgetA": errA_afterC - errA_afterA,
            "errC_beforeC": errC_beforeC, "errC_afterC": errC_afterC, "cbar": cbar}
```
> **Worker note:** if the fuse does NOT reduce forgetting vs always-plastic, the consolidation timescale is mistuned: A must consolidate (c↑, θ↓) during its `passes` before B/C arrive. Tune `tau_c`/`alpha_c`/`tau_w` (config knobs) so `cbar` is high after A, NOT the assertion. If the fuse causes plastic-death (errC_afterC not < errC_beforeC), `beta_c` (erosion on surprise) is too low — B/C are surprising so they must reopen θ. This knife-edge IS spec FM4; document the working values in concerns.

- [ ] **Step 4: PASS. Step 5: Commit** — `git commit -am "feat(stage3): continual A->B->C reconstruction harness + forgetting metric"`

---

## Task 4: EWC-analog baseline (quadratic anchor) on the same substrate

**Files:** Create `benchmarks/baselines/ewc.py`; add test to `tests/test_stage3_smoke.py`

The EWC comparator: after task A, store anchor weights `W*` and a diagonal importance `Ω` (a Fisher analog = accumulated `(Π·ε·a)²`), then add a quadratic penalty `−λ Ω (W − W*)` to the weight update during B,C. This is the "needs a Fisher pass + stored anchors" baseline GRAIL's fuse aims to match WITHOUT those.

- [ ] **Step 1: Failing test (append):**
```python
def test_ewc_baseline_runs_and_reduces_forgetting():
    from benchmarks.baselines.ewc import run_continual_ewc
    from benchmarks.tasks.continual import run_continual
    ewc     = run_continual_ewc(seed=0)
    plastic = run_continual(use_fuse=False, seed=0)
    assert ewc["forgetA"] < plastic["forgetA"]    # EWC also reduces forgetting (sanity: the task is learnable-retainable)
    assert ewc["used_fisher_pass"] is True         # EWC requires the extra importance pass GRAIL avoids
```

- [ ] **Step 2: Fail. Step 3: Implement** `benchmarks/baselines/ewc.py`:
```python
import numpy as np
from grail.config import GRAILConfig
from grail.pc_core import PCAreas
from grail.plasticity import Eligibility, weight_update, precision_update, feedback_update
from grail.neuromod import Neuromodulator
from grail.rng import SeededRNG
from benchmarks.tasks.continual import _prototypes, _err_on

def run_continual_ewc(seed=0, dim=10, per_task=6, passes=40, lam=5.0):
    cfg = GRAILConfig(dims=(dim,8), n_settle=10, seed=seed, tau_w=40.0)
    A,B,C = _prototypes(np.random.default_rng(seed+5), 3, per_task, dim)
    net = PCAreas(cfg); nm = Neuromodulator(cfg); rng = SeededRNG(seed)
    elig = [Eligibility((cfg.dims[l+1],), cfg) for l in range(net.L-1)]
    Wstar = [None]*(net.L-1); Omega = [np.zeros_like(net.W[l]) for l in range(net.L-1)]

    def train(patterns, anchored):
        for _ in range(passes):
            for p in patterns:
                for _ in range(cfg.n_settle): net.settle_step(rng, T=cfg.T_floor, clamp_bottom=p)
                net.compute_errors(); M = nm.update(reward=1.0)
                for l in range(net.L-1):
                    elig[l].step(a_pre=net.x[l+1])
                    dW = weight_update(M=M, theta=np.ones_like(net.W[l]), Pi_post=net.Pi[l],
                                       eps_post=net.eps[l], elig=elig[l].value, eta=cfg.eta_w/cfg.tau_w)
                    if anchored and Wstar[l] is not None:
                        dW = dW - (cfg.eta_w/cfg.tau_w)*lam*Omega[l]*(net.W[l]-Wstar[l])  # EWC quadratic penalty
                    net.W[l] += dW
                    net.B[l] += (1.0/cfg.tau_b)*feedback_update(net.B[l], a_up=net.x[l+1], eps=net.eps[l], cfg=cfg)
                    net.Pi[l] = precision_update(net.Pi[l], eps_sq=net.eps[l]**2, cfg=cfg)

    train(A, anchored=False)
    errA_afterA = _err_on(net, A, cfg, rng); errC_beforeC = _err_on(net, C, cfg, rng)
    # EWC's extra cost: a Fisher-importance pass over A + stored anchors (GRAIL's fuse needs neither)
    for p in A:
        for _ in range(cfg.n_settle): net.settle_step(rng, T=cfg.T_floor, clamp_bottom=p)
        net.compute_errors()
        for l in range(net.L-1):
            elig[l].step(a_pre=net.x[l+1])
            g = weight_update(M=1.0, theta=np.ones_like(net.W[l]), Pi_post=net.Pi[l],
                              eps_post=net.eps[l], elig=elig[l].value, eta=1.0)
            Omega[l] += g**2
    for l in range(net.L-1): Wstar[l] = net.W[l].copy()
    train(B, anchored=True); train(C, anchored=True)
    errA_afterC = _err_on(net, A, cfg, rng); errC_afterC = _err_on(net, C, cfg, rng)
    return {"errA_afterA":errA_afterA, "errA_afterC":errA_afterC, "forgetA":errA_afterC-errA_afterA,
            "errC_beforeC":errC_beforeC, "errC_afterC":errC_afterC, "used_fisher_pass":True}
```
- [ ] **Step 4: PASS. Step 5: Commit** — `git commit -am "feat(stage3): EWC-analog baseline (Fisher pass + anchors)"`

---

## Task 5: run_stage3 demonstration + README + full suite green

**Files:** Create `benchmarks/run_stage3.py`; update `README.md`; run full suite.

- [ ] **Step 1:** `benchmarks/run_stage3.py` (with the sys.path bootstrap) prints, averaged over seeds {0,1,2}: `forgetA` and `errC_afterC` for GRAIL-fuse, θ≡1 (always-plastic), and EWC-analog; plus GRAIL's `cbar` (consolidation reached).
```python
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from benchmarks.tasks.continual import run_continual
from benchmarks.baselines.ewc import run_continual_ewc

if __name__ == "__main__":
    seeds = (0,1,2)
    def avg(fn, **kw): 
        rs = [fn(seed=s, **kw) for s in seeds]; 
        return {k: float(np.mean([r[k] for r in rs])) for k in rs[0] if isinstance(rs[0][k],(int,float))}
    fuse = avg(run_continual, use_fuse=True); plastic = avg(run_continual, use_fuse=False); ewc = avg(run_continual_ewc)
    print(f"{'method':<16}{'forgetA':>10}{'errC_afterC':>14}")
    print(f"{'GRAIL-fuse':<16}{fuse['forgetA']:>10.3f}{fuse['errC_afterC']:>14.3f}   (cbar={fuse['cbar']:.2f})")
    print(f"{'always-plastic':<16}{plastic['forgetA']:>10.3f}{plastic['errC_afterC']:>14.3f}")
    print(f"{'EWC-analog':<16}{ewc['forgetA']:>10.3f}{ewc['errC_afterC']:>14.3f}   (+Fisher pass +anchors)")
```
- [ ] **Step 2:** Run `python3 benchmarks/run_stage3.py`, capture the table.
- [ ] **Step 3:** Update `README.md` Stage-3 section: the fuse reduces forgetting vs always-plastic WITHOUT replay/Fisher/anchors, competitive with EWC; reiterate honesty gate — OP3 is **addressed, not solved**; the (θ,c) loop is a knife-edge (FM4: forgetting OR plastic-death), tuned not proven; do NOT claim "stability-plasticity solved."
- [ ] **Step 4:** Full suite `cd /Users/nazmi/grail && python3 -m pytest -q` (all green).
- [ ] **Step 5: Commit** — `git commit -am "docs(stage3): README + catastrophic-forgetting demonstration; full suite green"`

---

## Self-Review (planner)

**Spec coverage:** §3③/§7 metaplastic fuse (per-synapse c, S̄, θ=σ(g(S−c)), reuses Π,ε,eligibility, no Fisher/anchors/task-boundary) → Task 2; gate-into-weight-update (θ from fuse multiplies the four-factor rule) → Task 3; §10 Task-2 sequential A→B→C, no replay/iid/boundary, forgetting metric + baselines (θ≡1, EWC-analog) → Tasks 3-4; FM4 knife-edge (forgetting vs plastic-death) tested both ways (forgetA reduced AND errC still learned) → Task 3 test + worker note; honesty gate (addressed-not-solved) → Task 5. ✓

**Placeholder scan:** complete code + behavioral tests throughout; worker notes flag the FM4 tuning knife-edge without leaving logic unspecified. ✓

**Type consistency:** `MetaplasticFuse(shape, cfg).update(Pi_post, eps_post, elig) -> theta` matches `weight_update(..., theta=theta, ...)` (theta is (out,in), same as `W[l].shape`); reuses Stage-1 `PCAreas`, `Eligibility((dims[l+1],), cfg)`, `weight_update/precision_update/feedback_update`, `Neuromodulator`, `SeededRNG` with current signatures. The continual harness's `_prototypes`/`_err_on` are shared by the EWC baseline via import. ✓

**Intentional scope gap:** Task-3 full energy/op learning-curves (instrumented since Stage 0) get a dedicated reporting pass later; Stage 3 focuses on the forgetting axis.
