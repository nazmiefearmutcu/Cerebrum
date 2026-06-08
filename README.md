# GRAIL — Grid-Referenced Annealed Inference with Local plasticity

GRAIL is a **predictive-coding, backprop-free, fully-local-plasticity, neuromorphic-targeted**
learning architecture, implemented in **pure NumPy** (no torch / jax / sklearn, no autograd).

Inference, routing, and learning are all noisy gradient descent on **one** free-energy functional
`F`, at three timescales. There is **no backpropagation, no weight transport, and no DFA** anywhere
in the `grail/` package — every weight, feedback, and precision update is a hand-written local rule.

> **Design spec:** [`docs/superpowers/specs/2026-06-08-grail-cortical-workspace-design.md`](docs/superpowers/specs/2026-06-08-grail-cortical-workspace-design.md)
> **Implementation plan (Stage 0+1):** [`docs/superpowers/plans/2026-06-08-grail-stage0-1-pc-core-grid-head.md`](docs/superpowers/plans/2026-06-08-grail-stage0-1-pc-core-grid-head.md)

This repository currently implements **Stage 0 + Stage 1**: the predictive-coding core (error
neurons, stochastic Langevin settling, four-factor local plasticity, separate feedback weights,
diagonal precision) plus the structured grid generative HEAD, validated on a TEM-class few-shot
graph-completion task. The gate / workspace / broadcast (Stage 2) and the metaplastic
stability-plasticity fuse (Stage 3) are **not yet built** — they are explicitly staged for later
plans.

---

## The five pillars

| Pillar | Mechanism in GRAIL |
|---|---|
| **1. Predictive-coding substrate** | Each cortical area `l` has a physically separate error-neuron population `ε_l = x_l − ŷ_l`. Inference = activities settling to minimize precision-weighted error. Errors flow, not raw activations. |
| **2. Fully-local plasticity** | Four-factor Hebbian `τ_w Ẇ = M·θ·Π·ε·e`; every factor is physically present at the synapse. The same `ε` that drives settling drives learning. |
| **3. Structured generative prior** | TEM-style grid×sensory factorization; frozen Lie-group rotation transitions driven by an **exogenous** action; the source of sample efficiency (graph-completion, not interpolation). |
| **4. Stochastic inference** | Langevin SDE settling `τ_x dx = −∂F/∂x dt + √(2τ_x T) dW`; samples an (approximate) posterior, never collapses to MAP (`T ≥ T_floor > 0`). |
| **5. Neuromorphic substrate** | Settling = analog device relaxation; intrinsic device noise = the Langevin floor; only the scalar `M` crosses the whole chip. (Load-bearing, with honestly-downgraded claims — see *Honest status* below.) |

---

## The bans — enforced as invariants in code

These are not style preferences; they are the line that separates GRAIL from backprop / DFA /
weight-transport methods. A violation invalidates the project. They are enforced as executable
assertions or structurally (see `grail/invariants.py`, `grail/types.py`, and the test suite).

1. **No backpropagation / no autograd** anywhere in `grail/`. Every update is a hand-written local
   rule. (The only exception is `benchmarks/baselines/backprop_mlp.py`, a clearly-labeled baseline
   *comparator* that is allowed manual backprop — it is not part of GRAIL.)
2. **No weight transport.** No update reads `Wᵀ`. Feedback uses a **separate** array `B`, an
   independent object updated by its own local rule (`grail/plasticity.py`).
3. **Scalar neuromodulator.** `M` is a scalar; no vector global signal ever enters a weight update
   (a vector global signal would be DFA).
4. **Exogenous `z_act`.** The grid transition driver is strictly exogenous:
   `GridHead.transition(...)` accepts **only** an `Exogenous(...)` wrapper; a plain (possibly
   data-derived) `ndarray` raises `TypeError`. `x`/`W`/gate are never wired into `z_act`.
5. **No sequence-mixer** (linear attention / delta rule / state-space / softmax attention).
6. **Success axis is sample efficiency** for this stage. No throughput / perplexity / latency claims.

---

## Honest status — what is and is NOT solved

GRAIL solves **zero** open problems. The architecture is a **bet**, and the riskiest part of that
bet is unproven.

| Open problem | Honest status |
|---|---|
| **Scaling** | **NOT solved — an UNPROVEN bet.** No fully-local, transport-relaxed, noisy-sampling method has matched backprop on hard tasks. With `B ≠ Wᵀ`, the rule does not even provably recover the true gradient at the fixed point. |
| **Backward-weight wart** | **Relaxed, not solved.** `B` replaces `Wᵀ` as a feedback-alignment-class approximation; transpose recovery is not guaranteed. |
| **Stability-plasticity** | **Genuinely addressed, NOT solved** (and not yet built — Stage 3). No stability proof; can fail toward catastrophic forgetting OR plastic-death. |
| **Global coherence** | **Pressured, not guaranteed.** |
| **Dead experts** | **Addressed, fragile in both directions.** No closed-form setpoint. |

**Explicit non-claims (these may NEVER be asserted about GRAIL):**

- ❌ No claim that **scaling is solved**. Scaling is an unproven bet; with `B ≠ Wᵀ` the update is
  not even a provable gradient.
- ❌ No claim that **stability-plasticity is solved**. The metaplastic fuse is not yet implemented
  and, even when built, carries no stability proof.
- ❌ No claim of **O(1) global communication**. Learn-time scalar comm is a *target*, not a proven
  property; infer-time broadcast/routing traffic is not O(1).

Because feedback uses `B ≠ Wᵀ`, the functional `F` as written is the **surrogate** vector field the
chip actually descends. The identity `F = −log p(x, g, data)` holds exactly only in the symmetric
limit `B = Wᵀ`; with `B ≠ Wᵀ` the drift is non-conservative — no scalar potential, no Lyapunov
guarantee. This is a stated failure mode, not a hidden one.

Losing to a transformer on GPU throughput is **expected and acceptable**. The only success axis
claimed here is **sample efficiency** on a structured relational task.

---

## Task-1 result — does the grid prior buy sample efficiency?

TEM-class few-shot **graph-completion**: fraction of *unobserved* graph edges correctly predicted
after `K ∈ {5, 10, 20}` observations, averaged over seeds `{0, 1, 2}` on a 4×4 gridworld with a
5-symbol vocabulary. The **load-bearing claim is grid > flat** (Pillar 3): the structured grid
prior must beat the flat positional prior, or Pillar 3 is not wired correctly. The backprop-MLP
column is reported for context but is **not** a headline claim.

Reproduce with `python3 benchmarks/run_task1.py`:

```
   K   GRAIL-grid   flat-prior   backprop-MLP
   5        0.503        0.258          0.258
  10        0.412        0.233          0.292
  20        0.312        0.245          0.260
```

GRAIL-grid beats the flat prior at every `K` (the Pillar-3 win). This is a small structured task,
not evidence of scaling — see *Honest status* above.

---

## Repository layout

```
grail/grail/        # the GRAIL package (pure NumPy, no autograd)
  config.py         # GRAILConfig — all hyperparameters
  rng.py            # SeededRNG — reproducible, zeroable noise
  types.py          # Exogenous wrapper (enforces z_act exogeneity by construction)
  invariants.py     # BAN-1/2/3 executable assertions
  counters.py       # energy/op + global-comm-event counters (LEARN vs INFER)
  nonlinear.py      # g_act = tanh and derivative
  pc_core.py        # PC areas: predictions, error neurons, diagonal precision, Langevin step
  plasticity.py     # eligibility traces, four-factor weight rule, feedback-B rule, precision rule
  neuromod.py       # scalar neuromodulator M and couplings
  grid_head.py      # structured grid prior: frozen modules, path integration, content store
  network.py        # GRAILCore (Stage 1: PC areas + grid HEAD, NO gate yet)
grail/tests/        # unit + invariant + load-bearing tests
grail/benchmarks/   # Task-1 task, baselines (flat-prior, backprop-MLP comparator), run_task1.py
```

---

## Running

```bash
cd grail
python3 -m pytest -q          # full test suite (no external deps beyond numpy)
python3 benchmarks/run_task1.py   # print the Task-1 result table
```

Python 3.11+, NumPy 2.x. No other dependencies. Nothing to install for the package itself.
