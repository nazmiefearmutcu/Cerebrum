# GRAIL — Session Handoff (resume from here)

**Last updated:** 2026-06-09 · **Repo:** `/Users/nazmi/grail` (git, local-only, no remote) · **61 commits · 159 tests green · ban-clean**

GRAIL = a predictive-coding, **backprop-free**, fully-local-plasticity, neuromorphic-targeted learning architecture, pure NumPy. Council-designed (13-agent scientific council + adversarial audit), then built in stages, then frontier-mapped via a ralph loop. It is the deliberate **anti-pattern** of the earlier PRISM-Seq escape to DeltaNet+DFA.

> Full design: `docs/superpowers/specs/2026-06-08-grail-cortical-workspace-design.md`
> Persistent memory: `~/.claude/projects/-Users-nazmi/memory/grail-project.md` (read it — it has the full blow-by-blow)

---

## LOCKED GOAL (do not drift)
Diverge from the transformer **only on brain-favorable axes**: (1) sample efficiency, (2) energy/ops, (3) online continual learning. **NEVER** GPU throughput / perplexity / latency — chasing those is the gravity well that killed PRISM. Pure NumPy, **no autograd**; GPU is "the microscope", not the target.

## ABSOLUTE BANS (enforced in code; a violation invalidates the project)
1. No backprop/autograd/torch/jax in `grail/` (allowed ONLY in labeled `benchmarks/baselines/` comparators).
2. No weight transport (no `W.T` read; feedback uses a separate `B` array).
3. Neuromodulator `M` is a SCALAR (no DFA vector into any weight update).
4. Workspace write strictly one-hot in `grail/` (soft only in `benchmarks/baselines/soft_mixer.py`).
5. Grid driver `z_act` strictly `Exogenous` (data-dependent z_act = ban).
6. The metaplastic fuse uses ONLY local Pi/eps/eligibility (no Fisher/anchors/task-boundary — those only in `benchmarks/baselines/ewc.py`).

## HONESTY GATE (the project's spine — never relax)
NEVER claim "scaling solved" / "stability-plasticity solved" / "O(1) global comm". Zero open problems are solved. A null/negative result, **mechanism-explained**, is a SUCCESS. Every headline number gets multi-seed 95% CIs + the right controls (untrained + random-projection). When a probe gives a bleak result, **CHECK THE PROBE before concluding the architecture fails** (item-5 was a degenerate task, not a model limit — see below).

---

## ARCHITECTURE (5 pillars, one free-energy F, three timescales)
- **Pillar 1 PC substrate:** error neurons `ε_l = x_l − ŷ_l`, stochastic Langevin settling.
- **Pillar 2 local plasticity:** four-factor Hebbian `τ_w Ẇ = M·θ·Π·ε·e`.
- **Pillar 3 structured prior:** frozen grid-cell HEAD (Lie-group rotations, exogenous path-integration) — the metric sample-efficiency lever.
- **Pillar 4 stochastic inference:** Langevin noise (`T_floor>0`).
- **Pillar 5 neuromorphic:** settling=device relaxation, scalar-M global comm.
All five wired together in `grail/unified.py` (`GRAILNet.step(obs_slices, action, reward)`).

## VERIFIED RESULTS (CI-honest)
| Axis | Result |
|---|---|
| Sample efficiency (grid prior) | metric graph: holds + widens to 16×16; transitive order: scales (N=25 GRAIL 1.0 vs MLP 0.63) |
| Emergent routing (no attention) | M=6 routing 0.81 (chance 0.167); soft-write ablation collapses to gated-SSM (participation 2.2) |
| Continual learning (no replay) | forgetA 0.055±0.039 vs always-plastic 0.557 (8/8 seeds, CI-separated); EWC-competitive without Fisher/anchors |
| **Local rule builds factored representation** | held-out factor decode **0.92** (chance 0.167), > untrained 0.825 + random-proj 0.85, CI-clean; systematic (holds under hard splits) |

## HONEST FRONTIER (where it BREAKS — all mapped, not hidden)
- Grid prior is a **metric** bias → BREAKS on non-metric/asymmetric directed graphs (FM7; backprop-MLP matches/beats GRAIL there).
- Continual fuse is a **budget-bounded** knife-edge → protection CI-separated only at ≤150 passes, breaks ≥200 (FM4).
- Pillar-4 settling noise is **NOT load-bearing for accuracy** (deterministic ≥ noisy; hurts continual) — only yields weak calibrated uncertainty (AUROC 0.64) at the native floor.
- "Learned-beyond-input" factorization margin runs out of headroom at cardinality ≥8 (concat input becomes linearly trivial).

---

## ⭐ THE SINGLE MOST IMPORTANT OPEN ISSUE (resume here)
**The factored latent and the frozen grid prior COMPETE in the full GRAILNet and only partially cooperate.**

- Probe (C3, `benchmarks/run_factorization_pipeline.py`): a bare module's factored latent (decode 0.92) **SURVIVES** +workspace-broadcast (0.91) and +metaplastic-fuse (0.91).
- The grid top-down originally **DESTROYED** it (decode 0.47; the grid Hebbian content-store prediction norm ~47 dominated the module top area, ~400× the latent |x|≈0.12).
- **FIXED** (merged, commit `0d2e7f4`): opt-in `GRAILConfig.balance_grid_precision=True` (PC-local precision-gain that down-weights a dominating top-down prediction to the bottom-up signal scale). → **+grid decode recovers 0.465→0.910**, grid few-shot **byte-identical**. The isolated competition is solved.
- **STILL OPEN (honest):** the **full `GRAILNet`** (grid+gate+workspace+fuse together) decode is only 0.11→0.28 with the fix on. **Refuted that it's under-training** (ran fast `eta=0.6` + 150 passes — does NOT recover, stays ~0.29 ≈ untrained). So the residual is a **deeper interaction** in `GRAILNet.step` — each piece *alone* preserves factorization (with the fix), the full integration still erases it.

### NEXT INVESTIGATION (exact)
Isolate which coupling in `grail/unified.py::GRAILNet.step` destroys the obs-driven factor code (it's NOT grid-domination, NOT learning budget):
1. The **broadcast-into-bottom during TRAINING** (does the workspace efference copy corrupt the bottom-area ε that drives plasticity?).
2. The **gate dynamics** (does one-hot selection / the bid loop inject combo-correlated noise into the latent?).
3. The **settle/learn order** in `step` vs the isolated probe's order.
Method: ablate each coupling in `run_factorization_pipeline.py`'s `full` path one at a time (a `+broadcast+grid` vs `+gate+grid` vs `+full` matrix), with `balance_grid_precision=True`, fast eta, multi-seed CIs + untrained/random-proj controls. Report which coupling is the disruptor + mechanism. If it can't be made to cooperate at this scale, that's a real architectural limit — report it honestly.

---

## HOW TO RESUME
- **Run all tests:** `cd /Users/nazmi/grail && python3 -m pytest -q` (python3 = 3.14, numpy 2.4.6; do NOT install anything).
- **Key benchmarks:** `python3 benchmarks/run_task1.py` (few-shot), `run_stage2.py` (routing), `run_stage3.py` (forgetting, 8-seed), `run_factorization.py` (the 0.92 factorization result), `run_factorization_pipeline.py` (the C3 open issue; set env `GRAIL_BALANCE_GRID_PRECISION=1` to enable the fix), `run_pillar4_ablation.py`, `run_uncertainty.py`, `run_scaling.py`.
- **The roadmap / loop task:** `RALPH_TASK.md` (P0 = the C3 cooperation issue above; then the deferred scaling roadmap).
- **Ralph loop status:** the scaling-frontier ralph loop (`completion_promise=GRAIL-SCALING-USER-APPROVED`) is **NOT currently active** (no working-tree `.claude/ralph-loop.local.md`). To resume sustained autonomous work, re-invoke `/ralph-loop:ralph-loop` pointing at `RALPH_TASK.md`. **Only the human emits the completion promise.**

## PROVEN BUILD METHOD (what worked here)
- council/spec → bite-sized TDD plan → **subagent-driven Workflow** (sequential implement → independent review + ban-audit + honesty-audit → fix per milestone). ~0 fix iters when the plan ships complete tests+code.
- For **hard/uncertain fixes**: 2-3 **parallel worktree-isolated** `Agent` subagents each try a different approach, controller picks the winner and merges (used for the central-bet null correction and the C3 fix).
- **Inside a ralph loop:** use **synchronous** parallel `Agent` subagents that finish within the turn (NOT background Workflows — the stop-hook races them); the controller serializes all git/README writes (one writer).
- Plans must ship COMPLETE runnable code+tests AND pre-verify numerics (literal code has had non-functional spots builders must catch).

## KEY FILES
```
grail/            config.py rng.py types.py invariants.py counters.py nonlinear.py
                  pc_core.py plasticity.py neuromod.py grid_head.py energy.py
                  gate.py workspace.py network.py network2.py metaplasticity.py unified.py
benchmarks/       run_task1/stage2/stage3/energy/uncertainty/pillar4_ablation/scaling.py
                  run_factorization{,_multi,_splits,_pipeline}.py
                  tasks/ (gridworld, graph_completion, binding, continual, compositional, relational, transitive...)
                  baselines/ (flat_prior, backprop_mlp, soft_mixer, ewc + per-task variants)
docs/superpowers/ specs/2026-06-08-grail-cortical-workspace-design.md  plans/*.md
RALPH_TASK.md     the sustained-dev roadmap (P0 = C3 cooperation)
```
