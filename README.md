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

The three staged prototypes are also wired together in **one coherent network**, `grail/unified.py`
(`GRAILNet`), whose single `step(obs_slices, action, reward)` exercises **all five pillars together**:
grid-HEAD path-integration on an exogenous action → cortical-module settling under both the grid
top-down and the workspace broadcast → scalar-bid stochastic one-hot gate / write / broadcast →
metaplastic-`θ`-gated four-factor local plasticity, all gated by the single scalar `M`. It composes
the existing Stage-1/2/3 modules (no logic duplicated) and is exercised by an end-to-end integration
test (`tests/test_unified.py`) pinning every invariant (one-hot write, scalar `M`, exogenous `z_act`,
`θ` actually gating module plasticity). It is an **integration** of the existing pieces — it does not
add a new headline result or change any Stage-1/2/3 number.

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

Reproduce with `python3 benchmarks/run_stage2.py` (mean ± 95% CI over 5 seeds). Two operating points,
two claims:

**2A — Routing (the gate at low selection temperature + Go-weight decay).** The target's salient object
gives it the strictly-highest scalar bid, so *argmax-bid* routing is in fact perfect (1.000); the cost
is that a hot, drifting gate throws much of that away. A low selection temperature (so the informative
bid dominates) plus a small Go/NoGo weight decay (so the gate does not learn spurious fixed preferences
on a randomly-rotating target) recover it — while staying a **stochastic** one-hot sample (Pillar 4):

```
[M=4] (chance=0.250)  one-hot routing_acc = 0.713 +/- 0.295  | win_entropy = 1.382 +/- 0.006
[M=6] (chance=0.167)  one-hot routing_acc = 0.806 +/- 0.200  | win_entropy = 1.779 +/- 0.014
```

Both clear chance with margin (`0.806±0.200` excludes 0.167), and high win-entropy confirms load stays
balanced (no dead/hog collapse — the reward-aware homeostasis no longer penalizes *correct* routing).

**2B — Write-rule ablation (one-hot vs soft, at a moderate temperature where `P` spreads).** A near-
degenerate low-temperature `P` makes the soft write approximate the one-hot write, hiding the effect; at
a moderate `gate_temp` the contrast is clean (matched temperature, only the write rule differs):

```
[M=4] one-hot routing = 0.616 +/- 0.253 (participation=1.0)  | soft routing = 0.276 +/- 0.188  participation = 2.20 +/- 0.31
[M=6] one-hot routing = 0.678 +/- 0.191 (participation=1.0)  | soft routing = 0.318 +/- 0.157  participation = 2.29 +/- 0.34
```

The forbidden soft aggregation blends **~2.2 modules per slot** (`participation` CI cleanly excludes 1.0)
and routes far worse than the one-hot write (`0.276` vs `0.616`, non-overlapping) — it has collapsed to a
content-gated continuous mixer (a gated-SSM / linear-attention/Mamba-class identity). **Strict one-hot
discreteness is load-bearing, not cosmetic.**

**Honesty gate (unchanged).** This stage still solves **zero** open problems. The architectural finding
is that GRAIL's no-query-key gate does **salience-driven + fixed-preference** routing (not content-
addressed routing — that would be attention, which is banned); on this small task the routing numbers
are a property of the bid signal + selection temperature, **not** evidence of scaling. Infer-time
broadcast traffic is **not** O(1); only the learn-time scalar `M` is.

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

Reproduce with `python3 benchmarks/run_stage3.py` (mean ± 95% CI over **8 seeds**, a **single fixed
knob set** — no per-task/per-seed retuning — and a **noise-free (T=0) measurement readout**; lower
`forgetA` is better):

```
method                       forgetA         errC_afterC
GRAIL-fuse           0.055 +/- 0.039     0.943 +/- 0.127   (cbar=0.93)
always-plastic       0.557 +/- 0.178     0.635 +/- 0.089
EWC-analog           0.109 +/- 0.047     0.864 +/- 0.140   (+Fisher pass +anchors)

robustness (8 seeds, single fixed knob set, T=0 noise-free eval):
  fuse < always-plastic on every seed : True (8/8)
  95% CIs separated (fuse upper 0.094 < plastic lower 0.379) : True
```

The fuse cuts mean forgetting to about **a tenth** of always-plastic (0.055 vs 0.557) — A consolidates
(`cbar≈0.93` after A, so `θ` closes before B/C arrive) — while still learning C (no plastic-death;
`errC_afterC` 0.943 < `errC_beforeC`). The result is now **robust across 8 seeds with a single fixed
knob set**: the fuse is lower on **every** seed and the 95% `forgetA` CIs are **cleanly separated**
(fuse upper bound 0.094 < always-plastic lower bound 0.379). It is **competitive with the EWC-analog**
(GRAIL-fuse forgetA 0.055 is even below EWC's 0.109) *without* EWC's Fisher pass or stored anchors —
EWC retains a small edge on C-learning (`errC_afterC` 0.864 vs 0.943).

**What changed vs the earlier overlapping-CI table (honest).** The fuse mechanism is **unchanged**;
the earlier `0.283 +/- 0.326` overlap was **measurement variance, not fuse variance**. `forgetA` was
read out by re-running the *stochastic* settling floor (`T_floor>0`) at eval time, which injects
~0.05-rms per-eval noise that dominated the cross-seed interval. The floor is a *learning-time*
regularizer (Pillar 4, forbids MAP collapse) — it has no business in a *measurement* of the already-
learned weights. Reading out deterministically (T=0, fresh fixed eval rng) cuts the fuse `forgetA`
cross-seed sd ~4-5× (≈0.21 → ≈0.047) and separates the CIs. Per-seed instrumentation confirms the
fuse itself was already robust (final `c.mean≈0.93` on every seed; the fuse beat always-plastic on
8/8 seeds even under the noisy eval). This is a measurement fix, **not** a fuse retune and **not** a
weakened assertion.

This still does **not** make stability-plasticity "solved" — it remains a tuned knife-edge with no
stability proof (FM4). Robustness is demonstrated at *these* knobs on *this* A→B→C task; it is **not**
a guarantee for arbitrary new tasks, harder streams, or knob settings.

**Honesty gate (critical).** OP3 (stability-plasticity) is **GENUINELY ADDRESSED — NOT SOLVED.** The
`(θ,c)` loop is a **tuned knife-edge**, not a proof. It is exactly spec failure-mode **FM4**, with **two**
ways to fall off: **catastrophic forgetting** (if `θ` never closes, A is overwritten) and **plastic-death**
(if `θ` never reopens, B/C cannot be learned). The numbers above hold at the working config knobs
(`tau_S, tau_c, alpha_c, beta_c, c_max, g_theta` in `grail/config.py`); there is **no stability proof** and
**no guarantee** of robustness to new tasks/seeds without tuning. **We do NOT claim "stability-plasticity
solved."** This stage solves **zero** open problems; it demonstrates **forgetting reduction without
replay/iid/Fisher/anchors**, which is the only success axis claimed here.

---

## Scaling probe (honest, exploratory)

> **This is an UNPROVEN BET, not a result.** Spec §7 open-problem #1 is explicit: *no
> fully-local, transport-relaxed, noisy-sampling method has matched backprop on hard tasks,
> and with `B ≠ Wᵀ` the rule does not even provably recover the true gradient at the fixed
> point.* The probe below runs the existing tasks at **larger sizes** and reports, with 95%
> CIs over 8 seeds, **exactly where the brain-axis advantages hold and where they break**. We
> make **no "scaling solved" claim.** Reproduce with `python3 benchmarks/run_scaling.py`.

### (a) Task-1 few-shot graph-completion on bigger gridworlds / larger vocab

Held-out edge-completion accuracy (mean ± 95% CI, 8 seeds; GRAIL-grid vs flat-prior vs the
backprop-MLP comparator):

| size | K | GRAIL-grid | flat-prior | backprop-MLP | chance |
|---|---|---|---|---|---|
| 4×4 v5  | 5  | **0.579 ± 0.187** | 0.161 ± 0.116 | 0.203 ± 0.165 | 0.200 |
| 4×4 v5  | 20 | **0.390 ± 0.085** | 0.228 ± 0.042 | 0.262 ± 0.116 | 0.200 |
| 6×6 v8  | 5  | **0.690 ± 0.179** | 0.141 ± 0.162 | 0.178 ± 0.189 | 0.125 |
| 6×6 v8  | 20 | **0.394 ± 0.121** | 0.156 ± 0.069 | 0.167 ± 0.064 | 0.125 |
| 8×8 v10 | 5  | **0.584 ± 0.149** | 0.068 ± 0.075 | 0.080 ± 0.079 | 0.100 |
| 8×8 v10 | 20 | **0.392 ± 0.070** | 0.126 ± 0.059 | 0.147 ± 0.055 | 0.100 |

**Verdict (honest):** the grid-prior advantage **does not shrink as the graph grows — it
holds, and the mean margin over the best baseline actually *grows* with grid size** (≈+0.38 at
4×4 K=5 → ≈+0.50 at 8×8 K=5), because flat-prior and the MLP both decay toward chance as the
graph gets larger while GRAIL still path-integrates unobserved edges. The advantage is **CI-
separated at every K on 8×8** and at low K on 6×6/4×4. What **shrinks is the margin as K rises**
(more observations let the baselines memorise more walked edges), so on the *small* 4×4 graph at
K=20 the CIs overlap — that is the expected "few-shot edge erodes with more data" boundary, **not**
a failure of the prior at scale. **Holds to 8×8 / vocab 10; few-shot edge narrows as K→20 on small
graphs.** (Still a small regime overall — no large-scale claim.)

### (b) Catastrophic forgetting with MORE sequential tasks (A→B→C→D→E)

Forgetting of the **first** task A (rise in its reconstruction error) measured after **each**
further task is learned, fuse vs always-plastic (mean ± 95% CI, 8 seeds; lower = better):

| after… | GRAIL-fuse | always-plastic |
|---|---|---|
| +1 task (B)         | **0.029 ± 0.031** | 0.435 ± 0.142 |
| +2 tasks (B,C)      | **0.055 ± 0.039** | 0.557 ± 0.178 |
| +3 tasks (B,C,D)    | **0.093 ± 0.052** | 0.531 ± 0.150 |
| +4 tasks (B,C,D,E)  | **0.107 ± 0.055** | 0.492 ± 0.111 |

**Verdict (honest):** the surprise-gated fuse **still protects the first task after four more
sequential tasks** — forgetting of A is **CI-separated below always-plastic at every step**, while
the fuse **still learns the last task** (last-task error drop 0.304 ± 0.078). First-task forgetting
**does creep up** (0.03 → 0.11 across +4 tasks) as the shared weights are repeatedly rewritten —
honest graceful degradation, not a cliff — and the consolidation reserve saturates (`c̄ ≈ 0.93`).
**Holds through A→…→E with the SAME fixed knob set (no per-task retuning).** No stability proof; a
much longer stream or a harder per-task overlap could still break it.

### (c) Deeper PC hierarchies on Task-1 (3–4 areas)

| depth (areas) | K=5 | K=10 | K=20 |
|---|---|---|---|
| 2 | 0.690 ± 0.179 | 0.503 ± 0.143 | 0.394 ± 0.121 |
| 3 | 0.690 ± 0.179 | 0.503 ± 0.143 | 0.394 ± 0.121 |
| 4 | 0.690 ± 0.179 | 0.503 ± 0.143 | 0.394 ± 0.121 |

**Verdict (honest):** adding PC areas has **no effect** on Task-1 — the numbers are identical
across depth. This is **mechanism-explained, not a bug**: Task-1 completion is driven by the
grid **HEAD** (path-integrated content store), so stacking more error-neuron areas adds inference
depth that the completion readout never consults. Deeper hierarchy **neither helps nor hurts** here;
a task whose structure actually requires hierarchical abstraction would be needed to probe whether
depth helps — that is future work, not a claim.

**Bottom line:** on these axes the brain-favorable advantages **hold at the larger sizes tested**
(grid-prior sample efficiency strengthens with graph size; fuse forgetting-protection survives five
sequential tasks), and **break / are null** exactly where expected (few-shot margin erodes with more
data on small graphs; depth is inert for a grid-head task). This is **honest evidence of where it
holds and breaks — not a scaling-solved claim.**

---

## Task-3 result — energy / operations (success axis 2)

GRAIL is event-driven: an error neuron only "spikes" (and drives its synapses) when its prediction
error exceeds threshold, so **dynamic switching energy decays as the network becomes competent**
(`ε → 0` ⇒ silent units ⇒ fewer synaptic ops). The learning signal that crosses the whole network is a
**single scalar** `M`, versus a backprop network's per-layer error **vector** (`O(depth)` elements).

Reproduce with `python3 benchmarks/run_energy.py` (reconstruction task; noise-free `T=0` measurement so
the metric reflects systematic error, not the Langevin noise floor):

```
 pass   recon_err eps_spars@0.1   dyn_ops  dyn_energy
    0      1.2787         0.833     133.3       47.12
   30      0.3840         0.633     101.3       26.08
  ...
  300      0.2847         0.633      96.0       22.68
```

As competence rises, **recon_err falls ~4.5×** and the magnitude-weighted **dynamic switching energy
falls ~2.1×**; the thresholded spike-sparsity@0.1 falls modestly (0.83 → 0.63). Meanwhile a matched
**dense backprop** net does **320 MAC ops/step with `ρ=1` (no decay)**, and its learn-time global
communication is **16 error-vector elements (`O(depth)`)** versus GRAIL's **1 scalar `M`**.

**Honesty gate.** Only the **dynamic** switching term decays — **static/leakage power and settle-time
energy do NOT** decay with competence, and the iterative settling can *cost more* steps precisely when
the posterior is interesting (spec FM2). The thresholded spike count is conservative (the local learner
plateaus at `recon ≈ 0.25`, so some units stay above the 0.1 threshold); the magnitude-weighted
`dyn_energy` tracks the true error decay. This is a small task — **not** a scaling or wall-clock claim,
and the infer-time broadcast traffic is **not** O(1).

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
  unified.py        # GRAILNet — ONE network exercising ALL FIVE pillars: grid HEAD path-integration (exogenous) → PC-module settling under grid top-down + workspace broadcast → scalar-bid one-hot gate/write/broadcast → metaplastic-θ-gated four-factor module plasticity + gate learning + reward-aware homeostasis, all gated by the single scalar M; composes the staged modules (no logic duplicated)
grail/tests/        # unit + invariant + load-bearing tests (incl. gate/workspace/network2/stage2-smoke, metaplasticity, stage3-smoke)
grail/benchmarks/   # Task-1 + Stage-2 binding task + Stage-3 continual A→B→C; baselines (flat-prior, backprop-MLP, soft-mixer ablation, EWC-analog); run_task1.py, run_stage2.py, run_stage3.py
  run_scaling.py    # I6 honest scaling probe: Task-1 on bigger grids/vocab, forgetting over A→B→C→D→E, deeper PC hierarchies — reports per-axis HOLDS/PARTIAL/BREAKS with CIs (UNPROVEN bet, no scaling-solved claim)
```

---

## Running

```bash
cd grail
python3 -m pytest -q            # full test suite (no external deps beyond numpy)
python3 benchmarks/run_task1.py    # print the Task-1 (grid prior / sample efficiency) result table
python3 benchmarks/run_stage2.py   # print the Stage-2 (emergent routing + one-hot-vs-soft) result table
python3 benchmarks/run_stage3.py   # print the Stage-3 (catastrophic-forgetting: fuse vs θ≡1 vs EWC) result table
python3 benchmarks/run_scaling.py  # I6 honest scaling probe: bigger grids, A→B→C→D→E forgetting, deeper hierarchies (per-axis HOLDS/BREAKS w/ CIs)
```

Python 3.11+, NumPy 2.x. No other dependencies. Nothing to install for the package itself.
