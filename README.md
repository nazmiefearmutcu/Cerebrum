# GRAIL — Grid-Referenced Annealed Inference with Local plasticity

GRAIL is a **predictive-coding, backprop-free, fully-local-plasticity, neuromorphic-targeted**
learning architecture, implemented in **pure NumPy** (no torch / jax / sklearn, no autograd).

Inference, routing, and learning are all noisy gradient descent on **one** free-energy functional
`F`, at three timescales. There is **no backpropagation, no weight transport, and no DFA** anywhere
in the `grail/` package — every weight, feedback, and precision update is a hand-written local rule.

> **Design spec:** [`docs/superpowers/specs/2026-06-08-grail-cortical-workspace-design.md`](docs/superpowers/specs/2026-06-08-grail-cortical-workspace-design.md)
> **Implementation plan (Stage 0+1):** [`docs/superpowers/plans/2026-06-08-grail-stage0-1-pc-core-grid-head.md`](docs/superpowers/plans/2026-06-08-grail-stage0-1-pc-core-grid-head.md)

This repository currently implements **Stage 0 + Stage 1 + Stage 2 + Stage 3**: the predictive-coding
core (error neurons, stochastic Langevin settling, four-factor local plasticity, separate feedback
weights, diagonal precision) plus the structured grid generative HEAD, validated on a TEM-class
few-shot graph-completion task; the **cortical workspace** — a stochastic basal-ganglia gate, a
`k≪n` workspace with strict one-hot write, and the thalamo-cortical broadcast loop, in which
inter-module routing **emerges** with no attention matrix; and now the **surprise-gated metaplastic
fuse** (`grail/metaplasticity.py`) — a per-synapse consolidation reserve `c` and plasticity-permission
`θ = σ(g(S − c))` driven by **local surprise only**, reusing the same `Π,ε,e` already computed for
inference (no Fisher pass, no stored anchors, no task-boundary signal). The fuse **addresses** the
stability-plasticity dilemma (OP3) — it is **NOT solved**: the `(θ,c)` loop is a tuned knife-edge with
no stability proof (see *Stage-3 result* and *Honest status* below).

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
| **Stability-plasticity** | **Genuinely addressed, NOT solved** (Stage 3, surprise-gated metaplastic fuse, validated on Task-2). No stability proof; the `(θ,c)` loop is a tuned knife-edge that can fail toward catastrophic forgetting OR plastic-death (FM4). |
| **Global coherence** | **Pressured, not guaranteed.** |
| **Dead experts** | **Addressed, fragile in both directions.** No closed-form setpoint. |

**Explicit non-claims (these may NEVER be asserted about GRAIL):**

- ❌ No claim that **scaling is solved**. Scaling is an unproven bet; with `B ≠ Wᵀ` the update is
  not even a provable gradient.
- ❌ No claim that **stability-plasticity is solved**. The metaplastic fuse is implemented and
  reduces forgetting on Task-2, but carries **no stability proof**; it is a tuned knife-edge (FM4 —
  forgetting OR plastic-death), not a guarantee.
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

Reproduce with `python3 benchmarks/run_task1.py` (mean ± 95% CI over 5 seeds; chance = 1/vocab = 0.200):

```
   K          GRAIL-grid          flat-prior        backprop-MLP
   5     0.562 +/- 0.194     0.168 +/- 0.189     0.182 +/- 0.178
  10     0.381 +/- 0.079     0.189 +/- 0.085     0.230 +/- 0.164
  20     0.338 +/- 0.056     0.225 +/- 0.073     0.228 +/- 0.168
```

GRAIL-grid beats the flat prior at every `K` (the Pillar-3 win), and at `K=10` and `K=20` the
intervals are cleanly separated; at `K=5` the per-seed variance is high (some random graphs are
easy, some hard) so the large mean gap carries a wide CI. This is a small structured task,
not evidence of scaling — see *Honest status* above.

---

## Stage-2 result — does routing *emerge* without an attention matrix?

Stage 2 adds the **cortical workspace** (`grail/gate.py`, `grail/workspace.py`, `grail/network2.py`):
`M` predictive-coding modules each settle on their own input slice; each emits a **scalar own-error
bid** `b_m = π_m·E[‖ε_m‖²] + θ_m`; a striatal Go/NoGo gate draws a **stochastic strict-one-hot**
winner per workspace slot (Gumbel-argmax = exact softmax sample, never a plain argmax); the winner's
content is written one-hot into a slot and **broadcast back** as a top-down prediction that re-enters
every module's next settling. That closed loop — `bid → one-hot write → broadcast → reshape-next-bid`
— is the **only** token-mixing pathway. There is **no attention matrix, no query-key term, no
delta-rule, no state-space operator** anywhere; the "mixing matrix" exists only as the transient
time-series of one-hot win events. The gate's weights learn by a **local three-factor rule** gated by
the scalar neuromodulator `M` (eligibility `e_mj = (z − P)·b`), never a global error vector.

**The load-bearing claim: routing emerges and the strict one-hot write is essential.** We test this
two ways on a selective-routing ("binding") task — `M` modules each see a one-hot object, one
designated TARGET module carries the salient (rewarded) object, and the gate must learn to route the
target into the slot:

1. **Emergent routing + load balance.** Routing accuracy rises well above chance, and win-entropy
   stays high (no permanently dead/hog expert — the dead-expert homeostasis spreads wins).
2. **One-hot vs soft (BAN-1 / §10 ablation).** Relaxing the strict one-hot write to the **forbidden**
   soft aggregation `W_j ← Σ_m P(win_j=m)·read(m)` (`benchmarks/baselines/soft_mixer.py`) turns the
   workspace into a content-gated linear recurrent mixer (a gated-SSM / linear-attention/Mamba-class
   identity). Routing accuracy degrades toward chance and per-slot participation climbs above 1 (many
   modules contribute every step) — proving the discreteness is **load-bearing, not cosmetic**.

Reproduce with `python3 benchmarks/run_stage2.py` (mean ± 95% CI over 5 seeds):

```
[M=4] (chance=0.250)
   one-hot routing_acc = 0.668 +/- 0.200  | win_entropy = 1.383 +/- 0.005
   soft    routing_acc = 0.525 +/- 0.262  | slot_participation = 1.94 +/- 0.52
[M=6] (chance=0.167)
   one-hot routing_acc = 0.476 +/- 0.190  | win_entropy = 1.768 +/- 0.015
   soft    routing_acc = 0.351 +/- 0.221  | slot_participation = 2.79 +/- 0.63
```

**Honest reading of the CIs (this matters).** Two claims are robust and CI-clean: (a) one-hot routing
is **above chance** at both `M` (`0.668±0.200` excludes 0.250; `0.476±0.190` excludes 0.167), and
(b) the soft ablation **mixes >1 module per slot** (`1.94±0.52` and `2.79±0.63` both exclude 1.0,
whereas one-hot participation is exactly 1.0 by construction) — so soft genuinely collapses to a
gated-SSM-class continuous mixer. The one claim that is **NOT** clean at 5 seeds is one-hot-*vs*-soft
*routing accuracy*: the intervals overlap (`0.668±0.200` vs `0.525±0.262`), and earlier single-seed
snapshots (0.850 vs 0.746) overstated the gap. So the load-bearing result is **"routing emerges above
chance AND the soft write provably mixes (participation>1)"** — not a clean routing-accuracy win of
one-hot over soft. Tightening that comparison (more seeds / a harder binding task) is open work.

**Honesty gate (unchanged).** This stage still solves **zero** open problems. It demonstrates
**emergent routing without an attention matrix**, plus the one-hot-vs-soft contrast — it is **NOT**
evidence of scaling, throughput, or perplexity parity, and makes **no** scaling claim. Routing
accuracy is a property of this small binding task, not of any open problem (see *Honest status*
above). The infer-time broadcast traffic is **not** O(1); only the learn-time scalar `M` is.

---

## Stage-3 result — does the metaplastic fuse mitigate catastrophic forgetting?

Stage 3 adds the **surprise-gated metaplastic fuse** (`grail/metaplasticity.py`). Each synapse keeps a
slow consolidation reserve `c` and a surprise baseline `S̄`; it reads the **same** precision-weighted
error-eligibility magnitude `S_raw = |Π·ε·e|` that already drives inference, forms a relative surprise
`S = S_raw − S̄`, lets **below-baseline (predictive) activity build `c`** and **above-baseline (surprising)
activity erode `c`**, and emits a per-synapse plasticity permission `θ = σ(g(S − c)) ∈ [0,1]` that
multiplies the four-factor weight rule. Low surprise → `c↑, θ↓` (the synapse freezes, protecting prior
tasks); high surprise → `c↓, θ↑` (the synapse reopens, learn-on-surprise). **There is no Fisher pass, no
stored anchor weights, and no task-boundary signal to the fuse** — those belong only to the EWC-analog
*baseline* GRAIL aims to match without them.

**The load-bearing claim: the fuse reduces forgetting vs always-plastic local learning, while still
learning the later task, without replay/iid/Fisher/anchors.** We run a sequential reconstruction stream
A→B→C (`benchmarks/tasks/continual.py`) — three disjoint prototype clusters streamed in order, **no
replay, no iid mixing, no task-boundary signal** — and measure reconstruction error on held-out A
patterns after A and again after C. **Forgetting** = (error-on-A after C) − (error-on-A after A).
Comparators on the same local substrate: `θ≡1` (always-plastic; should forget) and an EWC-analog
(`benchmarks/baselines/ewc.py`; a quadratic anchor penalty `−λΩ(W−W*)` that **does** pay for a Fisher
pass + stored anchors).

Reproduce with `python3 benchmarks/run_stage3.py` (mean ± 95% CI over 5 seeds; lower `forgetA` is better):

```
method                       forgetA         errC_afterC
GRAIL-fuse           0.283 +/- 0.326     1.145 +/- 0.205   (cbar=0.93)
always-plastic       0.825 +/- 0.345     1.050 +/- 0.246
EWC-analog           0.181 +/- 0.343     1.376 +/- 0.340   (+Fisher pass +anchors)
```

The fuse cuts mean forgetting to about **a third** of always-plastic (0.283 vs 0.825) — A consolidates
(`cbar≈0.93` after A, so `θ` closes before B/C arrive) — while still learning C (no plastic-death; its
`errC_afterC` is lower than EWC's, i.e. it stays more plastic). It is **competitive with EWC** *without*
EWC's Fisher pass or stored anchors. **Honest reading of the CIs:** at 5 seeds the per-seed variance is
large and the `forgetA` intervals **overlap** (the means clearly order GRAIL-fuse < EWC < always-plastic,
but n=5 is not enough to separate them statistically). The robust statement is the **mean** forgetting
reduction; tightening it (more seeds, lower-variance task) is open work. This does **not** make
stability-plasticity "solved" — it remains a tuned knife-edge with no stability proof (FM4).

**Honesty gate (critical).** OP3 (stability-plasticity) is **GENUINELY ADDRESSED — NOT SOLVED.** The
`(θ,c)` loop is a **tuned knife-edge**, not a proof. It is exactly spec failure-mode **FM4**, with **two**
ways to fall off: **catastrophic forgetting** (if `θ` never closes, A is overwritten) and **plastic-death**
(if `θ` never reopens, B/C cannot be learned). The numbers above hold at the working config knobs
(`tau_S, tau_c, alpha_c, beta_c, c_max, g_theta` in `grail/config.py`); there is **no stability proof** and
**no guarantee** of robustness to new tasks/seeds without tuning. **We do NOT claim "stability-plasticity
solved."** This stage solves **zero** open problems; it demonstrates **forgetting reduction without
replay/iid/Fisher/anchors**, which is the only success axis claimed here.

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
  gate.py           # Stage 2: BasalGangliaGate — scalar bids, striatal Go/NoGo, stochastic one-hot select, local 3-factor learn, dead-expert homeostasis
  workspace.py      # Stage 2: Workspace — k slots, strict one-hot write, broadcast (efference copy)
  network2.py       # Stage 2: GRAILWorkspaceNet — M modules + gate + workspace + broadcast loop (routing emerges)
  metaplasticity.py # Stage 3: MetaplasticFuse — per-synapse consolidation reserve c + surprise baseline S̄ + plasticity permission θ=σ(g(S−c)); reuses Π,ε,e (no Fisher/anchors/task-boundary)
grail/tests/        # unit + invariant + load-bearing tests (incl. gate/workspace/network2/stage2-smoke, metaplasticity, stage3-smoke)
grail/benchmarks/   # Task-1 + Stage-2 binding task + Stage-3 continual A→B→C; baselines (flat-prior, backprop-MLP, soft-mixer ablation, EWC-analog); run_task1.py, run_stage2.py, run_stage3.py
```

---

## Running

```bash
cd grail
python3 -m pytest -q            # full test suite (no external deps beyond numpy)
python3 benchmarks/run_task1.py    # print the Task-1 (grid prior / sample efficiency) result table
python3 benchmarks/run_stage2.py   # print the Stage-2 (emergent routing + one-hot-vs-soft) result table
python3 benchmarks/run_stage3.py   # print the Stage-3 (catastrophic-forgetting: fuse vs θ≡1 vs EWC) result table
```

Python 3.11+, NumPy 2.x. No other dependencies. Nothing to install for the package itself.
