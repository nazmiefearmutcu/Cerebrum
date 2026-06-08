# GRAIL — Ralph sustained task: SCALING FRONTIER (probe the central unproven bet, honestly)

**Mission:** Push GRAIL (`/Users/nazmi/grail`, 89 tests green) onto HARDER tasks and LARGER scales to **map honestly where its brain-axis advantages hold and where they break**. This directly attacks the central UNPROVEN bet (spec §7 OP1: no fully-local method has matched backprop at scale) and the spec's own flagged failure modes (esp. **FM7**: frozen commuting rotation-blocks may fail on non-metric / asymmetric / abstract relational graphs). The deliverable is **honest evidence of the frontier**, NOT a "scaling solved" claim.

## How this loop runs (speed + safety)
- **Each iteration, do the work via a BACKGROUND Workflow** (multi-agent, fast) — NOT slow single-threaded inline edits.
- **Concurrency rule (critical):** if a GRAIL background workflow is still running (check `/workflows` / TaskList), do **NOT** start another and do **NOT** edit the repo this turn — briefly note progress and yield. Only launch the next workflow when none is running. Never have two writers on the repo.
- After a workflow completes: independently verify (run suite + re-run the new benchmark + ban-audit), integrate/commit if needed, update README, then launch the next item.

## ABSOLUTE BANS (unchanged — a violation invalidates the project)
No backprop/autograd/torch/jax in `grail/` (only in labeled `benchmarks/baselines/`); no weight transport (`W.T`); scalar `M` only; strict one-hot workspace write in `grail/`; exogenous-only `z_act`; the metaplastic fuse uses ONLY local `Π/ε/eligibility` (no Fisher/anchors/task-boundary — those only in `ewc.py`); success only on sample-efficiency / energy-ops / continual-learning, never throughput/perplexity/latency.

## HONESTY GATE (unchanged)
NEVER claim "scaling solved" / "stability-plasticity solved" / "O(1) global comm". Report where it BREAKS as loudly as where it holds. A NULL or NEGATIVE result, mechanism-explained, is a SUCCESS for this loop. Multi-seed CIs on every headline number. Don't weaken assertions to pass; downgrade claims instead.

## ROADMAP (priority order — highest unfinished first)
1. **Non-metric / asymmetric relational graphs (probe FM7).** Build a relational task whose transitions do NOT compose as commuting rotations: directed/asymmetric graphs, trees/hierarchies, abstract (non-Euclidean) relations. Compare GRAIL-grid vs flat-prior vs backprop-MLP few-shot. HONEST expectation: the rotation-block grid prior should DEGRADE here (its algebra assumes metric/Euclidean composition). Map exactly where and how much it breaks vs the metric gridworld where it shines.
2. **Transitive inference** (classic relational generalization): a total order A>B>C>D>E; observe only ADJACENT pairs; test NON-adjacent pairs (e.g. B vs D). Does the grid/structured prior generalize the ordering from few examples better than baselines?
3. **Larger metric graphs:** push Task-1 from 8×8 to 12×12, 16×16 (and bigger vocab). Does the few-shot grid advantage hold, shrink, or break with size? CIs.
4. **Harder continual learning:** more tasks and/or class-incremental / sharper boundaries (beyond the A→B→C reconstruction stream). Does the metaplastic fuse still reduce first-task forgetting, and where does the knife-edge break, without per-task retuning?
5. **Deeper-hierarchy fix:** the I6 depth null-result was because Task-1 completion reads only the grid HEAD. Build a COMPOSITIONAL task where deeper PC areas are actually consulted, and test whether depth then helps (or honestly does not).
6. **Frontier synthesis:** a single honest "where GRAIL holds / breaks" map across all axes + a README "Scaling frontier" section with CIs and explicit non-claims.

## Method per item
For each item, the background Workflow should: write/extend tests (TDD), implement the harder task + run it with multi-seed CIs, run the FULL suite green + a ban-audit, commit, and update the README. Independent review agent verifies (re-run + ban + honesty audit) before the item is considered done.

## Loop termination
Do NOT emit the completion promise `GRAIL-SCALING-USER-APPROVED` yourself under ANY circumstances — only the human stops this loop. Keep mapping the frontier until they do.
