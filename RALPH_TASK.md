# GRAIL — Ralph sustained-development task

**Mission:** Advance and harden GRAIL (the predictive-coding, backprop-free, fully-local-plasticity, neuromorphic brain architecture at `/Users/nazmi/grail`) on its three brain-favorable axes: sample efficiency, energy/ops, online continual learning. This is the deliberate anti-pattern of a GPU-friendly sequence mixer — do NOT drift toward DeltaNet/attention/SSM.

**Each iteration, FIRST read:** `docs/superpowers/specs/2026-06-08-grail-cortical-workspace-design.md` and `README.md` for full context. Then pick the SINGLE highest-priority unfinished roadmap item below, implement it strictly test-first (TDD), run the FULL pytest suite to green (never regress the existing tests — currently 59), run a ban-audit grep over `grail/`, and commit with a clear message.

## ABSOLUTE BANS (never violate — a violation invalidates the project)
1. NO backprop / autograd / torch / jax anywhere in the `grail/` package. Every update is a hand-written local rule. Allowed ONLY in clearly-labeled `benchmarks/baselines/` comparators.
2. NO weight transport: no update may read the transpose of a forward weight (`W.T`). Feedback uses a SEPARATE `B` array.
3. The neuromodulator `M` is a SCALAR. No vector global signal may enter any weight update (that is DFA).
4. The workspace write is STRICT one-hot. The soft aggregation `W_j = sum_m P*read(m)` is FORBIDDEN in `grail/` — it exists only in `benchmarks/baselines/soft_mixer.py` as the labeled ablation.
5. The grid driver `z_act` is strictly `Exogenous` — a data-dependent `z_act` is a ban (selective-SSM).
6. Success is measured ONLY on sample-efficiency / energy-ops / continual-learning. NEVER GPU throughput, latency, or perplexity.

## HONESTY GATE (never violate)
- NEVER claim "scaling solved", "stability-plasticity solved", or "O(1) global comm".
- Zero open problems are solved. Scaling is an unproven bet. The metaplastic fuse is a tuned knife-edge with no stability proof.
- Report failure modes and where things break. Do not overclaim. If a result weakens, say so.

## ROADMAP (priority order — do the highest unfinished one)
1. **Clean the Stage-3 fuse consolidation drive** in `grail/metaplasticity.py`: it currently double-counts `relu(-S)` via two identical terms (`max(S_bar - S_raw, 0)` and `max(-S, 0)` are equal since `S = S_raw - S_bar`). Refactor into ONE clear predictive-regime drive, and re-verify the forgetting table still holds (GRAIL `forgetA` < always-plastic `forgetA`, and GRAIL still learns C). Re-run `benchmarks/run_stage3.py`.
2. **Multi-seed confidence intervals + sensitivity sweeps** in `benchmarks/run_task1.py`, `run_stage2.py`, `run_stage3.py` so every headline number carries a CI over >=5 seeds, and key knobs get a small sensitivity sweep.
3. **Task-3 ENERGY/OP LEARNING CURVES** from the already-instrumented `Counters`: show dynamic switching-energy decaying with competence (as `eps -> 0` and sparsity `rho -> floor`); report learn-time O(1) scalar-M vs infer-time O(k*T_settle) broadcast traffic vs a dense backprop baseline; HONESTLY report that static/leakage and settle-time energy do NOT necessarily decay. Add tests + a `run_energy.py`.
4. **Strengthen weak spots:** raise the M=6 routing accuracy (currently 0.354 vs chance 0.167) and make the catastrophic-forgetting knife-edge hold across seeds WITHOUT per-task retuning of `(tau_c, alpha_c, beta_c, g_theta)`.
5. **Unify** Stage-1 `GRAILCore` + Stage-2 `GRAILWorkspaceNet` + Stage-3 metaplastic fuse into ONE coherent network class exercising all five pillars together, with an integration test.
6. **Explore the UNPROVEN scaling bet** on harder relational + continual tasks (bigger graphs, more tasks/classes, deeper hierarchies), framed honestly as a bet — report where it holds and where it breaks.

## Method
For any non-trivial item, prefer the proven build method: write a bite-sized TDD micro-plan, then implement and independently review (run the full suite + ban-audit grep) before committing. Keep files focused. Update `README.md` results tables when numbers change.

## Loop termination
Do NOT emit the completion promise `GRAIL-ADVANCED-USER-APPROVED` yourself under ANY circumstances. Only the human stops this loop. Keep iterating and improving until they do.
