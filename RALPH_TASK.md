# CEREBRUM — Ralph sustained task: SCALING FRONTIER (probe the central unproven bet, honestly)

**Mission:** Push CEREBRUM (`/Users/nazmi/cerebrum`, 89 tests green) onto HARDER tasks and LARGER scales to **map honestly where its brain-axis advantages hold and where they break**. This directly attacks the central UNPROVEN bet (spec §7 OP1: no fully-local method has matched backprop at scale) and the spec's own flagged failure modes (esp. **FM7**: frozen commuting rotation-blocks may fail on non-metric / asymmetric / abstract relational graphs). The deliverable is **honest evidence of the frontier**, NOT a "scaling solved" claim.

## How this loop runs (speed + safety — learned the hard way)
- **Each iteration, do the heavy work via PARALLEL `Agent` subagents that COMPLETE WITHIN THE TURN** (fast multi-agent, but synchronous). Do **NOT** use background Workflows inside this loop — the ralph stop-hook re-fires on every turn-end and would race a still-running background workflow (concurrent repo writers + wasted no-op yield turns).
- **Concurrency rule (critical):** the controller (you) serializes all git: subagents CREATE/EDIT distinct new files and REPORT numbers, but do **NOT** commit and do **NOT** edit shared files (README, shared run scripts). After the subagents return, YOU verify (full suite + re-run benchmark + ban-audit), update README, and commit — one writer.
- Give each parallel subagent a DISTINCT file set so they never collide.

## ABSOLUTE BANS (unchanged — a violation invalidates the project)
No backprop/autograd/torch/jax in `cerebrum/` (only in labeled `benchmarks/baselines/`); no weight transport (`W.T`); scalar `M` only; strict one-hot workspace write in `cerebrum/`; exogenous-only `z_act`; the metaplastic fuse uses ONLY local `Π/ε/eligibility` (no Fisher/anchors/task-boundary — those only in `ewc.py`); success only on sample-efficiency / energy-ops / continual-learning, never throughput/perplexity/latency.

## HONESTY GATE (unchanged)
NEVER claim "scaling solved" / "stability-plasticity solved" / "O(1) global comm". Report where it BREAKS as loudly as where it holds. A NULL or NEGATIVE result, mechanism-explained, is a SUCCESS for this loop. Multi-seed CIs on every headline number. Don't weaken assertions to pass; downgrade claims instead.

## CURRENT TOP PRIORITY (Phase 2 — make the architecture better, not just mapped)
**P0 — Resolve the C3 lever-competition (the single most important open issue).** Probes showed the LOCAL rule builds a compositionally-generalizing factored latent (held-out decode 0.92, systematic), but in the UNIFIED CerebrumNet the FROZEN grid prior DOMINATES the module's top area (grid content-store prediction norm ~47 vs bare latent |x|~0.12, ~400x) and ERASES the factorization (full-CerebrumNet decode collapses to chance 0.11). The two sample-efficiency levers COMPETE instead of COOPERATE. FIX it (cerebrum/-touching, opt-in flag, default-unchanged, all 156 tests green): make factorization SURVIVE in the full CerebrumNet (decode recover from 0.11 toward ~0.9) WITHOUT breaking the grid prior's few-shot graph-completion win. Candidate mechanisms: (a) bound/decay the grid Hebbian content-store norm so its prediction doesn't dominate; (b) route grid top-down and the learned factor code to SEPARATE areas/latent subspaces; (c) precision-balance the two predictions. Try several in parallel worktree-isolated subagents, pick the winner, verify BOTH metrics (factorization-in-full-CerebrumNet AND grid few-shot still works). Honest: if they cannot be made to cooperate at this scale, report that as a real architectural limit.

## ROADMAP (later, after P0)
1. **Non-metric / asymmetric relational graphs (probe FM7).** Build a relational task whose transitions do NOT compose as commuting rotations: directed/asymmetric graphs, trees/hierarchies, abstract (non-Euclidean) relations. Compare CEREBRUM-grid vs flat-prior vs backprop-MLP few-shot. HONEST expectation: the rotation-block grid prior should DEGRADE here (its algebra assumes metric/Euclidean composition). Map exactly where and how much it breaks vs the metric gridworld where it shines.
2. **Transitive inference** (classic relational generalization): a total order A>B>C>D>E; observe only ADJACENT pairs; test NON-adjacent pairs (e.g. B vs D). Does the grid/structured prior generalize the ordering from few examples better than baselines?
3. **Larger metric graphs:** push Task-1 from 8×8 to 12×12, 16×16 (and bigger vocab). Does the few-shot grid advantage hold, shrink, or break with size? CIs.
4. **Harder continual learning:** more tasks and/or class-incremental / sharper boundaries (beyond the A→B→C reconstruction stream). Does the metaplastic fuse still reduce first-task forgetting, and where does the knife-edge break, without per-task retuning?
5. **Deeper-hierarchy fix:** the I6 depth null-result was because Task-1 completion reads only the grid HEAD. Build a COMPOSITIONAL task where deeper PC areas are actually consulted, and test whether depth then helps (or honestly does not).
6. **Frontier synthesis:** a single honest "where CEREBRUM holds / breaks" map across all axes + a README "Scaling frontier" section with CIs and explicit non-claims.

## Method per item
For each item, the background Workflow should: write/extend tests (TDD), implement the harder task + run it with multi-seed CIs, run the FULL suite green + a ban-audit, commit, and update the README. Independent review agent verifies (re-run + ban + honesty audit) before the item is considered done.

## Loop termination
Do NOT emit the completion promise `CEREBRUM-SCALING-USER-APPROVED` yourself under ANY circumstances — only the human stops this loop. Keep mapping the frontier until they do.
