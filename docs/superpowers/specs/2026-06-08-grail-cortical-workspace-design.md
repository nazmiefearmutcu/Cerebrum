# GRAIL — Grid-Referenced Annealed Inference with Local plasticity, over a Cortical Workspace

**Design spec — 2026-06-08**
Status: APPROVED for implementation (staged core, pure NumPy, no autograd).
Produced by a 13-agent scientific council (6 domain designers → chief-architect synthesis → 5-lens adversarial audit → revision). Audit found 2 critical + 8 major + 22 total violations; all addressed in revision.

---

## 0. What this is (and what it is NOT)

GRAIL is a predictive-coding, **backprop-free**, **fully-local-plasticity**, **neuromorphic-targeted** learning architecture. It is a single temperature-controlled free-energy machine in which **inference, routing, and learning are all noisy gradient descent on ONE functional `F`** at three timescales.

**It is NOT** a polished sequence-mixer (DeltaNet/Mamba/linear-attention), NOT trying to beat the transformer on GPU throughput or large-scale perplexity, and NOT using DFA. Those are the gravity well that killed the previous attempt (PRISM-Seq → Gated-DeltaNet + DFA). The success axis is **fixed** to brain-favorable axes only:

1. **Sample efficiency** — generalize from 5–20 examples.
2. **Energy / operations** — cost-per-op, count of global-communication events, activation sparsity (NOT GPU clock).
3. **Online continual learning** — catastrophic-forgetting over sequential tasks A→B→C, no iid batch, no replay.

Losing to the transformer on GPU throughput is **expected and acceptable**.

### Interpretation of the top-level goal
The user's opening line ("dramatically surpass the transformer") is reconciled with the embedded spec ("do NOT try to beat the transformer on GPU/perplexity") as: **dramatic divergence on the brain-favorable axes** (sample efficiency, energy, continual learning), where the substrate is right — not on GPU throughput, where the substrate is wrong and chasing it drags the design back to DeltaNet.

---

## 1. The five pillars and how GRAIL satisfies each

| Pillar | Mechanism in GRAIL |
|---|---|
| **1. Predictive-coding substrate** | Each cortical area `l` has a physically separate error-neuron population `ε_l = x_l − ŷ_l`. Inference = activities settling to minimize precision-weighted error. Errors flow, not raw activations. |
| **2. Fully-local plasticity** | Four-factor Hebbian `τ_w Ẇ = M·θ·Π·ε·e`; every factor physically present at the synapse. The same `ε` that drives settling drives learning. |
| **3. Structured generative prior** | TEM-style grid×sensory factorization; frozen Lie-group rotation transitions driven by **exogenous** action; the source of sample efficiency (graph-completion, not interpolation). |
| **4. Stochastic inference** | Langevin SDE settling `τ_x dx = −∂F/∂x dt + √(2τ_x T) dW`; samples an (approximate) posterior, never collapses to MAP (`T ≥ T_floor > 0`). |
| **5. Neuromorphic substrate** | Settling = analog device relaxation; intrinsic device noise = the Langevin floor; only the scalar `M` crosses the whole chip. (Load-bearing, with honestly-downgraded claims — see §7.) |

---

## 2. The single free-energy functional `F`

Over activities `x = {x_l}`, grid latent `g`, workspace `W = {W_1..W_k}`, gate assignments `z`, and all weights:

```
F = Σ_l [ ½ ε_lᵀ Π_l ε_l − ½ log det Π_l ]    (1) layered precision-weighted error + normalizer (Π_l DIAGONAL — load-bearing for locality)
  + ½ ‖ε^g‖²_{Σ_g⁻¹}                           (2) structural (grid) transition error
  + (−log p(g_0))                              (3) structured prior boundary term (HEAD)
  + R(x) = Σ_l γ‖x_l‖_1                         (4) activity cost / sparsity
  + λ Σ_j H_slot(W_j)                           (5) workspace slot regularizer (binding capacity)
  − β H_gate(z)                                 (6) gate entropy (load-balancing / anti-dead-expert)
```

**Term definitions**
- `ε_l = x_l − ŷ_l`, top-down prediction `ŷ_l = g_act(W_l x_{l+1})`, `W_l` the FORWARD generative synapse, `f = g_act'`. For the lowest areas `ŷ_l` also receives the workspace broadcast `p_l = Σ_j D_{lj} W_j` (efference copy), so `ε_l = x_l − ŷ_l − p_l`. **Critical separation:** the broadcast `p_l` enters ONLY inference, as a PREDICTION; it NEVER appears in any weight update. The only quantity crossing into a weight update from outside the synapse is the scalar `M`. If the broadcast ever fed `dW` as a vector, that would be DFA — it does not.
- `Π_l` = area `l`'s precision, a slow-learned local **DIAGONAL** gain. Diagonal is load-bearing: a full-covariance `−½ log det Π` normalizer needs a matrix inverse (non-local); diagonal keeps every factor single-unit-local.
- `ε^g = g − D(z_act,t) g_{t−1}`, the grid structural error. `z_act` is **strictly exogenous** (see §5 invariant).
- `W_j` = the `k ≪ n` workspace slot latents; `H_slot` keeps slots distinct.
- `z = {z_{mj}}` = sampled, strictly one-hot gate assignments; `−β H_gate(z)` rewards spreading wins across modules.

**Surrogate-energy disclosure (honesty):** because feedback uses `B ≠ Wᵀ` (§3, ban-3 relaxation), `F` as written is the **surrogate** vector field the chip actually descends. The identity `F = −log p(x,g,data)` (Bogacz 2017) holds **exactly only** in the symmetric limit `B = Wᵀ`. With `B ≠ Wᵀ` the drift is non-conservative — no scalar potential, no Lyapunov guarantee. This is a stated failure mode, not hidden.

---

## 3. The three update rules (all gradient descent on the same `F`)

Timescale ordering: `τ_ε ≪ τ_x ≪ τ_gate ≪ τ_W`.

### ① Activity settling — fast, stochastic (Langevin SDE)
```
τ_x dx_l = [ −Π_l ε_l
             + B_{l−1}( f'(·) ⊙ Π_{l−1} ε_{l−1} )        ← feedback via SEPARATE synapse B, NOT Wᵀ
             + Σ_j D^fb_{lj}( precision · ε^wksp )         ← workspace broadcast (seed of emergent mixing)
             − ∂R/∂x_l ] dt
           + √(2 τ_x T) dW_l(t),   dW_l ~ N(0, I dt)
```
- Term 2 is the **weight-transport relaxation** (BAN-3 disclosure): error relayed to the ONE adjacent area through physically separate `B_{l−1}`. Adjacent-only → not a global error vector → not DFA. Reciprocal-wiring assumption: `B_{l,ji}` and `W_{l,ij}` connect the SAME neuron pair.
- Grid latent settles on the same SDE: `τ_g dg = [ ε^g·B_g + Σ_l V_l(Π_l ε_l) ] dt + √(2 τ_g T) dW_g` (`B_g`, `V_l` separate feedback weights; `Σ_l V_l(·)` is a local-to-hub fan-in, not a chip-wide broadcast).
- Error neuron relaxation: `τ_ε dε_l/dt = (x_l − ŷ_l) − Π_l⁻¹ ε_l` → `ε` encodes the precision-weighted residual at equilibrium.

### ② Gate / routing — medium, stochastic, locally learned (PBWM striatal Go/NoGo)
```
Bid (only scalar leaving a module):  b_m = π_m · E_post[‖ε_m‖²] + θ_m
Striatal Go:  u_mj = G_mj b_m − Σ_{m'≠m} N_{m'j} b_{m'}     (NoGo lateral inhibition, local to slot-j pool)
Selection:    P(win_j=m) = softmax_m( u_mj / T_gate + ξ_mj ),  ξ ~ Gumbel/membrane noise
              z_mj = ONE-HOT SAMPLE      (NEVER argmax, NEVER soft-weighted)
Learning:     e_mj = (z_mj − P(win_j=m)) · b_m              (REINFORCE-as-three-factor eligibility)
              ΔG_mj = +η_G · M · e_mj    (Go)
              ΔN_mj = −η_N · M · e_mj    (NoGo opponent)
Homeostasis:  θ_m ← θ_m + γ_up(1 − Σ_j z_mj) − γ_dn Σ_j z_mj   (dead-expert term, rises on loss)
```
- **Bid invariant:** `b_m` is a SCALAR own-error salience from the module's own error-posterior moments only. NO cross-module content-similarity, NO query-key dot product anywhere → the competition can never become attention.
- `T_gate` set by the SAME scalar `M` (neuromodulator as gate inverse-temperature).

### ③ Weight plasticity — slow, local, four-factor Hebbian
```
τ_w dW_{l,ij}/dt = M(t) · θ_{l,ij} · Π_{l,i} · ε_{l,i} · e_{l,ij}
```
- `M(t)` = global SCALAR neuromodulator `= r − r̄`, `τ_r dr̄/dt = −r̄ + r`. The ONLY non-local signal (one diffuse wire). Sets when/sign; cannot assign credit alone.
- `θ_{l,ij} ∈ [0,1]` = metaplastic gate/fuse (§4).
- `Π_{l,i}` = postsynaptic diagonal precision (the SAME Π in `F`; precision = plasticity-rate identity). **Enters exactly ONCE** (precision-once convention; eligibility is the bare presynaptic low-pass) so `−∂F/∂W = −Π·ε·a` holds exactly.
- `ε_{l,i}` = postsynaptic error-neuron activity (same ε that drives settling).
- `e_{l,ij}` = synapse-local eligibility, `τ_e ė = −e + a_{l+1,j}` (bare presynaptic low-pass).

**Feedback synapse `B`** — local Kolen-Pollack-type rule on the same neuron pair: `τ_B Ḃ_l = η_B a_{l+1} ε_lᵀ − λ_B B_l`. **Honest downgrade:** because `W`'s update is gated by `M`,`θ` and carries `Π` while `B`'s is raw `a·ε` with decay, the updates are NOT matched → KP co-alignment is NOT guaranteed. Claim downgraded to "feedback-alignment-class approximation; transpose recovery not guaranteed."

**Precision learning (diagonal):** `τ_Π Π̇_{l,i} = −(Π_{l,i} − Π_0) + κ(Π_{l,i}⁻¹ − ⟨ε_{l,i}²⟩)` → `Π → 1/(σ0² + ⟨ε²⟩)`.

---

## 4. Stochastic term & temperature

Three noise sources:
1. **Langevin settling noise** `√(2τ_x T) dW_l` — converts deterministic MAP settling into approximate posterior sampling. **Honest identity:** stationary density `∝ exp(−F_surrogate / T_eff)`, equals the true posterior ONLY if `B → Wᵀ` AND the chain mixes — neither guaranteed → biased, finite-time sample.
2. **Plasticity programming jitter** `σ_prog(G) ζ` added to `ΔW` (weight-space exploration).
3. **Gumbel/membrane noise** `ξ_mj` making gate selection a one-hot sample.

**Temperature — two cleanly separated roles:**
- **Noise FLOOR (free, fixed):** intrinsic device thermal noise supplies fixed `T_floor > 0`. Its only job is to forbid MAP collapse. We do NOT claim `D = k_B T_eff/C` sets the model temperature to the posterior `T = 1`; it sets the floor only.
- **Annealing + M-coupling (active, NOT free):** `T(t) = T_floor + (T_0 − T_floor) e^{−t/τ_anneal}`, surprise reset `T_0 ← T_floor + κ|M|`, and couplings `Π = Π^base σ(a_Π M)`, `T = T_floor + b_T relu(M)`, `η = η_0 relu(M)`, `T_gate ∝ 1/M`. Implemented by an ACTIVE controlled noise source (gated current-mode noise cell), explicitly NOT harvested thermal physics.

---

## 5. Structured generative prior (the HEAD) — source of sample efficiency

```
p(g_{0:T}, w_{0:T}, x_{0:T} | z_act,{0:T}) = p(g_0) ∏_t p(g_{t+1}|g_t, z_act,t) · p(w_t|g_t,M_t) · p(x_t|w_t)
Transition: p(g_{t+1}|g_t,z_act) = N(g_{t+1}; D(z_act)g_t, Σ_g)
            D(z_act) = blockdiag_m R_m(z_act),  R_m(z_act) = exp(z_act Ω_m),  Ω_m = ω_m J,  J = [[0,−1],[1,0]]
Multi-frequency tiling: λ_m ≈ λ_0 r^m  (r ≈ 1.4),  ω_m geometric
Continuity/objectness: ‖R_m(z_act) − I‖ ≤ κ‖z_act‖;  p(w_t|w_{t−1}, same-object) = N(w_t; w_{t−1}, σ_obj² I)
Content store (fast Hebbian, M-gated):  M_t = M_{t−1} + η(w_t ⊗ ĝ_t);  completion ŵ_t = φ(M_t g_t);  ŷ_t = U ŵ_t
```

**NON-NEGOTIABLE INVARIANT (BAN-1):** `z_act` is **strictly exogenous** (action/motor efference). It is NEVER a function of tokens `x`, workspace `W`, gate `z`, modules, or any learned readout of state. If any data-dependent path feeds `z_act`, the grid latent becomes a data-dependent selective linear recurrent state = Mamba/selective-SSM = a BAN-1 violation. **Enforced as a code assertion/ablation** (zero gradient/connectivity from `x,W,z` to `z_act`).

The rotational recurrence `g_t = R(z_act)g_{t−1}` is acceptable ONLY because (i) `R` is frozen after development and (ii) its driver is exogenous → a fixed action-integrated reference frame, not a data-conditioned mixer. It path-integrates over **egomotion**, never over token content.

**Why sample efficiency (and nowhere else):** a flat prior must relearn transition structure from `O(states²)` observations. The grid prior ALREADY IS the transition structure, so a new environment needs only `O(states)` bindings into `M_t`; the rest is path-integrated over exogenous actions. From 5–20 observations the machine fills unobserved states by composing rotations + reading `M_t` — generalization by **graph-completion**. Structure transfers, content rebinds. Structural weights `D, ω_m, U` frozen after development; only `M_t` adapts within an episode.

---

## 6. Novelty — honest tally (borrowed vs new)

GRAIL is a new **synthesis** + ~2 thin candidate-novelties + 1 novel local instantiation + 1 novel architectural claim. **ZERO open problems fully solved.**

**Genuinely / thinly NEW:**
1. *[thin]* **Sampling-temperature folded into the neuromodulatory field** — identifying the Langevin posterior-sampling temperature of the inference SDE with the same scalar `M` that sets precision/gate-temp/learning-rate, so exploration-noise co-varies with attention/routing/plasticity. (Caveat: if even this leg has precedent, it reduces to pure synthesis.)
2. *[novel local instantiation]* **Surprise-gated per-synapse metaplastic fuse `θ/c`** — the consolidation signal REUSES the same `ε` already computed for inference (no Fisher pass, no task-boundary, no stored anchor weights, no quadratic penalty); a fuse that OPENS on surprise rather than a penalty that resists change; event-driven.
3. *[novel synthesis / architectural claim]* **Routing-as-disinhibition-recurrence as the SOLE mixer** — the closed loop [scalar bid → noisy striatal one-hot WTA → one-hot write → broadcast as top-down prediction → reshapes next bids] is the ONLY token-mixing pathway. The "mixing matrix" exists only as the transient time-series of one-hot win events.
4. *[thin]* **Posterior-moment bidding** — gate + weight rule consume first AND second moments of the noisy sample trajectory, so uncertainty reaches routing and learning for free.

**Borrowed (credited):** F+ε+Π (Bogacz/Rao-Ballard/Friston); Langevin neural sampling (Buesing 2011, Aitchison & Lengyel); learned feedback to dodge transport (Kolen-Pollack, Akrout); three-factor eligibility (Frémaux & Gerstner); grid×sensory factorization + grid-as-path-integrator (TEM, Whittington et al.; Gao, Sorscher); PBWM striatal Go/NoGo + REINFORCE-as-three-factor + neuromodulator-as-gate-temp (O'Reilly & Frank; Daw, Frank); global-workspace ignition/broadcast (Dehaene); EqProp physical-equilibrium-as-inference + device-noise-as-resource (Scellier-Bengio, Kendall-Scellier); single scalar setting precision AND learning rate (Yu & Dayan 2005, Doya 2002; Aston-Jones & Cohen adaptive-gain); cascade consolidation reserve (Benna-Fusi 2016, Fusi 2005); Bayesian synapse (Aitchison et al. 2021).

---

## 7. Open problems — honest status

| # | Problem | Status |
|---|---|---|
| 1 | **Scaling** | **NOT solved — unproven bet.** No fully-local, transport-relaxed, noisy-sampling method has matched backprop on hard tasks. With `B ≠ Wᵀ` the rule does not even provably recover the true gradient at the fixed point. |
| 2 | **Backward-weight wart** | **Relaxed, not solved.** `B` replaces `Wᵀ` as a feedback-alignment-class approximation; transpose recovery not guaranteed. |
| 3 | **Stability-plasticity** | **Genuinely addressed (NOT solved), pending Task-2 validation.** Surprise-gated metaplastic fuse `θ/c`; no stability proof; can fail to forgetting OR plastic-death (FM4). |
| 4 | **Global coherence** | **Pressured, not guaranteed.** Only the broadcast loop enforces it; may be too weak (Frankenstein latent risk). |
| 5 | **Dead experts** | **Addressed both-directions-fragile.** Local homeostatic `θ_m` + `−β H_gate`; no closed-form setpoint (FM5b). |

---

## 8. Differentiation (point-by-point)

- **vs Predictive Coding:** GRAIL adds Langevin noise (samples, not MAP), replaces `Wᵀ` with separate `B`, embeds PC energy as only term 1 of a larger F with frozen grid prior + stochastic gate + entropy terms, couples precision to the shared neuromodulator.
- **vs EqProp** (strawman removed): one noisy phase (no nudged phase) + scalar-gated Hebbian + structured grid prior + stochastic gate, vs two deterministic contrastive phases. GRAIL also samples a biased approximate posterior (surrogate energy) — honest.
- **vs Forward-Forward:** opposite — errors flow, activities settle, predictions are generative top-down, inference is recurrent + stochastic (FF is feedforward, two forward passes, no error neuron).
- **vs Target Prop:** GRAIL never computes/propagates a target vector; only local `ε` + scalar `M` cross synapses. No inverse networks.
- **vs DFA** (the key distinction): DFA broadcasts a global error VECTOR through a random matrix to every layer's weight update — banned. GRAIL broadcasts only a SCALAR `M` into weight updates; all spatial credit is the locally-settled `ε`. The feedback `B` relays error to one adjacent area during INFERENCE (not the learning signal). "If M became a vector this would be DFA" is the line we refuse to cross.

---

## 9. Failure modes (acknowledged, not hidden)

1. **Surrogate-energy / no-Lyapunov bias** (`B ≠ Wᵀ`) — non-conservative drift; no convergence guarantee; biased posterior even with perfect mixing.
2. **Mixing time vs settling budget** — nonconvex multimodal surrogate; short window → soft-MAP masquerading as posterior; inflates energy/sample.
3. **Precision/plasticity unification can starve novelty** — `η ∝ Π` means high-uncertainty regions are both ignored AND never learned (chicken-and-egg). Needs a `Π`-floor or transient `M`-override (partially breaks the clean single-scalar claim).
4. **Metaplastic knife-edge** (why OP3 is "addressed", not "solved") — `(θ,c,S̄)` loop hysteretic, no closed-form stability; mistuned → catastrophic forgetting OR plastic-death; strong noise inflates `ε²` → read as surprise → prevents consolidation.
5a. **Gate credit-assignment variance** — scalar-M REINFORCE through a sampled discrete gate is high-variance; may need many samples, trading against the sample-efficiency axis we must win.
5b. **Dead-experts / load-balancing fails both directions** — too little pressure → dead experts; too much → noise into slots; no closed-form setpoint; interacts with `T_gate`.
6. **Global coherence not guaranteed** — separately-settling modules writing shared slots can form a Frankenstein latent.
7. **WTA / frozen-prior expressivity ceilings** — `k` one-hot slots may be too coarse for fine relational binding (and we've banned the soft relaxation, so this ceiling is structural); frozen commuting rotation-blocks may fail on non-metric/asymmetric relational graphs; passive (no-action) settings remove the transition driver.
8. **Hardware realism + split** — the τ_RC analog-settling property and the event-driven/scalar-comm + per-synapse metaplastic state live on **mutually exclusive** hardware (analog crossbar at 10³–10⁴ devices vs Loihi-class digital that does not do analog equilibrium). No single chip does both at scale. Memristor noise is non-white/non-stationary → even the "free floor" may sample a wrong distribution.
9. **Self-flagged contradictions:** (a) "fully local" is true only up to ONE global scalar `M` (standard Frémaux-Gerstner compromise); (b) "physics does the settling for free" holds only for the symmetric sub-network; (c) two sites are local-to-a-microcircuit (slot-j WTA pool; grid hub fan-in), not strictly single-synapse; (d) BAN-1 compliance is conditional on never relaxing the one-hot write to soft weights and never letting `z_act` become data-dependent — both stated as invariants whose violation = a ban breach.

---

## 10. Prototype recipe — the RIGHT axes only

**Substrate:** GPU/CPU SIMULATION only (Euler-Maruyama for the SDEs); pure NumPy, NO autograd, all local rules written by hand. GPU is the microscope, never the success axis. Report nothing about tokens/sec or perplexity; report latency only as a physics property excluded from any transformer head-to-head.

### Task 1 — Few-shot generalization (does the grid prior buy sample efficiency?)
TEM-class relational task: structured gridworld / graph navigation + a small relational-analogy set (transitive-inference chains; 5-way/5-shot Omniglot-style bind-then-complete). **Metric:** fraction of UNOBSERVED graph edges correctly predicted after `K ∈ {5,10,20}` observations (graph-completion score). **Baselines:** (a) GRAIL with grid HEAD, (b) GRAIL with flat/additive positional prior (ablates Pillar 3), (c) small backprop MLP/transformer on the same K. **Win:** GRAIL-grid beats both flat-prior and backprop at K=5–20 on graph-completion; losing at large K is acceptable. **BAN-1 enforcement:** automated check that `z_act` has ZERO learned/data-dependent connectivity from `x,W,z`.

### Task 2 — Catastrophic forgetting (does the metaplastic fuse work — the OP3 contingency)
Sequential stream A→B→C, NO iid batch, NO replay, NO task-boundary signal (split-MNIST/split-Omniglot or sequential gridworld rooms). **Metric:** accuracy on A after learning B,C (backward transfer); forward accuracy on C; plot `θ̄`,`c̄` over time. **Baselines:** GRAIL with `θ≡1` (always-plastic → should forget), GRAIL with EWC-style global penalty, online SGD. **Win:** GRAIL retains A within a small drop while still learning C, WITHOUT replay, beating always-plastic and matching/beating EWC without its Fisher pass — AND holding without per-task retuning of `(α,β,τ_c)`.

### Task 3 — Operations / energy counter (instrument, not GPU clock)
Per-step counters:
- (i) `N_glob` as TWO numbers — LEARN-time (target O(1) scalar `M`) AND INFER-time (O(k·T_settle) broadcast vectors + O(n) bids/routing step); compare infer-time traffic against the backprop simulator's traffic.
- (ii) `ρ(t)` activation sparsity as a learning curve (competence-decaying DYNAMIC energy `ρ → ρ_floor`).
- (iii) TRUE energy/sample = settle-steps-to-mixing × per-step synaptic-ops (`ρ·FANIN·N`), during learning AND at convergence.
- (iv) STATIC/leakage power estimate (independent of `ρ`) + energy of per-synapse analog-state updates (`e, Π, S̄, c`).
- (v) settle-time-to-tolerance (physics property, not a success metric).
**Win:** GRAIL shows O(1) LEARN-time scalar comm + a decreasing dynamic-energy-per-sample curve (report the TOTAL-energy curve honestly — static + settling do not necessarily decay). Backprop shows O(depth) vector broadcasts + dense MAC (`ρ=1`).

### Ablations (each pillar/invariant is load-bearing)
- `T_floor → 0` (kills Pillar 4 → expect dead experts / over-confidence / forgetting).
- flat prior (kills Pillar 3 → expect Task-1 collapse).
- `θ ≡ 1` (kills the forgetting fix → expect Task-2 collapse).
- gate → argmax (kills stochastic routing → expect dead experts in Task-3 plots).
- **soft-weight-mixing relaxation** (BAN-1 enforcement): replace strict one-hot write with `W_j ← Σ_m P(win_j=m)·read(m)` and SHOW it degrades to a gated-SSM identity + metrics shift toward the SSM baseline → proves discreteness is load-bearing.
- dead-expert load-balance sweep: vary `β`, `θ_m` gain to show failure in BOTH directions.

---

## 11. Implementation staging (approved: staged core, pure NumPy)

**Stage 0 — Scaffolding & invariant harness.** Repo layout; the two BAN-1 invariants as first-class executable checks (one-hot write assertion; `z_act` exogeneity assertion); energy/op counter instrumentation; test harness; honesty/falsifiability test suite (lever-off==identity-style invariants where applicable).

**Stage 1 — PC + plasticity + stochastic core + grid HEAD → Task-1.** Hierarchical PC areas with error neurons, Langevin settling (Euler-Maruyama), four-factor local plasticity, separate feedback weights `B` (no transport), diagonal precision `Π`, and the grid generative HEAD. Run Task-1 few-shot with the flat-prior and backprop baselines.

**Stage 2 — Gate + workspace + broadcast (the emergent mixer) → multi-module routing.** Basal-ganglia stochastic one-hot gate, `k≪n` workspace, thalamo-cortical broadcast. Demonstrate emergent routing WITHOUT an attention matrix; run the soft-weight-mixing ablation showing collapse to SSM.

**Stage 3 — Metaplasticity (surprise-gated `θ/c` fuse) → Task-2.** Add the consolidation reserve + fuse; run Task-2 catastrophic-forgetting with `θ≡1`, EWC, online-SGD baselines.

**Throughout — Task-3 energy/op accounting** instrumented from Stage 0.

After the staged core is validated, sustained improvement is handed to a ralph-loop, forward-gated: no "scaling solved" / "stability-plasticity solved" / "O(1) global comm" claim may ever be emitted (they are misrepresentations per §6–§7).

---

## 12. Non-negotiable invariants (enforced in code)

1. **One-hot write:** slot content is `W_j ← read(argsample)`. `W_j ← Σ_m P(win)·read(m)` is FORBIDDEN (BAN-1). Asserted.
2. **Exogenous `z_act`:** zero learned/data-dependent connectivity from `x,W,z` into `z_act` (BAN-1). Asserted.
3. **Scalar neuromodulator:** `M` is a scalar; no vector global signal enters any `dW` (BAN-2/DFA). Asserted.
4. **No weight transport:** no update reads `Wᵀ`; feedback uses separate `B` (BAN-3). Structural.
5. **No autograd / no backprop:** all updates are local rules written by hand (BAN-2). Structural.
6. **Success axis:** only sample-efficiency, energy/ops, continual learning are reported as success; latency/throughput/perplexity are excluded (BAN-5).
