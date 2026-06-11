"""I6 — Scaling probe (HONEST, EXPLORATORY).

This harness runs the existing CEREBRUM tasks at LARGER sizes and reports, with
confidence intervals over seeds, WHERE the brain-axis advantages HOLD and WHERE
they BREAK. It is explicitly an UNPROVEN BET (spec §7 open-problem #1): no
fully-local, transport-relaxed, noisy-sampling method has matched backprop on
hard tasks. The deliverable is honest evidence, NOT a "scaling solved" claim.

Three axes:
  (a) probe_task1_scaling   — few-shot graph-completion on bigger gridworlds /
      larger vocab; CEREBRUM-grid vs flat-prior vs backprop-MLP across seeds.
  (b) probe_forgetting_scaling — catastrophic-forgetting with MORE sequential
      tasks (A->B->C->D->E); does the surprise-gated fuse still protect the
      FIRST task after several more, or does it break?
  (c) probe_depth_scaling   — deeper PC hierarchies (3-4 areas) on Task-1.

BAN COMPLIANCE: everything in this file routes through the existing cerebrum/
primitives. No backprop in CEREBRUM paths (the MLP is the labeled comparator),
scalar M only, strict one-hot is irrelevant (no workspace here), z_act is
Exogenous (graph_completion drives the grid via Exogenous moves), and the
N-task forgetting loop reuses ONLY local Pi/eps/eligibility surprise in the
fuse — NO Fisher pass, NO stored anchors, NO task-boundary signal.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

import numpy as np

from cerebrum.config import CerebrumConfig
from cerebrum.core_net import CerebrumCore
from cerebrum.pc_core import PCAreas
from cerebrum.plasticity import Eligibility, weight_update, precision_update, feedback_update
from cerebrum.metaplasticity import MetaplasticFuse
from cerebrum.neuromod import Neuromodulator
from cerebrum.rng import SeededRNG

from benchmarks.stats import mean_ci
from benchmarks.tasks.gridworld import make_episode
from benchmarks.tasks.graph_completion import run_cerebrum_episode
from benchmarks.baselines.flat_prior import run_flat_episode
from benchmarks.baselines.backprop_mlp import run_mlp_episode

# Continual-learning knobs: reuse the SAME fixed knob set the Stage-3 harness
# settled on (see benchmarks/tasks/continual.py). We do NOT retune per-task or
# per-n_tasks — that is the honest, hard test (spec Task-2 win condition).
from benchmarks.tasks.continual import (
    PROTO_SCALE, LATENT, ETA_W, TAU_W, TAU_E, TAU_R,
    TAU_C, ALPHA_C, BETA_C, G_THETA, TAU_S, _err_on,
)


def _stat(raw):
    """Pack a per-seed list into {'mean','ci','raw'} (95% CI half-width over seeds)."""
    m, h = mean_ci(raw)
    return {"mean": float(m), "ci": float(h), "raw": [float(x) for x in raw]}


# ---------------------------------------------------------------------------
# (a) Task-1 few-shot graph-completion at larger sizes
# ---------------------------------------------------------------------------
def probe_task1_scaling(sizes=((4, 4, 5), (6, 6, 8), (8, 8, 10)),
                        Ks=(5, 10, 20), seeds=(0, 1, 2, 3, 4),
                        n_settle=10, mlp_epochs=80):
    """For each (h,w,vocab) size and each K, score CEREBRUM-grid vs flat vs MLP.

    Returns out[(h,w,vocab)][K] = {cerebrum|flat|mlp: {mean,ci,raw}, 'chance': 1/vocab}.
    The K observations are held fixed across the three methods (same episode).
    """
    out = {}
    for (h, w, vocab) in sizes:
        out[(h, w, vocab)] = {}
        for K in Ks:
            g, f, m = [], [], []
            for s in seeds:
                ep = make_episode(h=h, w=w, vocab=vocab, K=K, seed=s)
                # grid head sized generously so larger graphs still path-integrate;
                # latent areas scale modestly with vocab so the decode has capacity.
                lat = max(8, 2 * vocab)
                cfg = CerebrumConfig(dims=(vocab, lat, lat), grid_n_modules=8,
                                  n_settle=n_settle, seed=s)
                g.append(run_cerebrum_episode(CerebrumCore(cfg), ep))
                f.append(run_flat_episode(ep))
                m.append(run_mlp_episode(ep, epochs=mlp_epochs, seed=s))
            out[(h, w, vocab)][K] = {
                "cerebrum": _stat(g), "flat": _stat(f), "mlp": _stat(m),
                "chance": 1.0 / vocab,
            }
    return out


# ---------------------------------------------------------------------------
# (b) Catastrophic forgetting with MORE sequential tasks (A->B->C->D->E)
# ---------------------------------------------------------------------------
def _make_cfg_continual(seed, dim):
    return CerebrumConfig(
        dims=(dim, LATENT), n_settle=10, seed=seed,
        tau_w=TAU_W, eta_w=ETA_W, tau_e=TAU_E, tau_r=TAU_R,
        tau_c=TAU_C, alpha_c=ALPHA_C, beta_c=BETA_C, g_theta=G_THETA, tau_S=TAU_S,
    )


def _continual_n_tasks(use_fuse, n_tasks, seed=0, dim=10, per_task=6, passes=100):
    """Generalized A->B->...->(n_tasks) continual stream on the SAME local substrate
    as benchmarks/tasks/continual.py, but recording the forgetting of the FIRST task
    after EACH subsequent task is learned. No Fisher/anchor/task-boundary — pure local
    surprise in the fuse.

    Returns:
      forget_first[m] = err(A) after m further tasks - err(A) right after A  (m=1..n_tasks-1)
      learn_last      = err(last task) after learning it  (drop from its pre-train err)
      cbar            = mean consolidation reserve at the end (fuse only)
    """
    cfg = _make_cfg_continual(seed, dim)
    rng_proto = np.random.default_rng(seed + 5)
    tasks = [[PROTO_SCALE * rng_proto.standard_normal(dim) for _ in range(per_task)]
             for _ in range(n_tasks)]
    net = PCAreas(cfg); nm = Neuromodulator(cfg); rng = SeededRNG(seed)
    elig = [Eligibility((cfg.dims[l + 1],), cfg) for l in range(net.L - 1)]
    fuse = [MetaplasticFuse(net.W[l].shape, cfg) for l in range(net.L - 1)] if use_fuse else None

    def train(patterns):
        for _ in range(passes):
            for p in patterns:
                for _ in range(cfg.n_settle):
                    net.settle_step(rng, T=cfg.T_floor, clamp_bottom=p)
                    for l in range(net.L - 1):
                        elig[l].step(a_pre=net.x[l + 1])
                net.compute_errors()
                M = nm.update(reward=1.0)
                for l in range(net.L - 1):
                    theta = (fuse[l].update(net.Pi[l], net.eps[l], elig[l].value)
                             if use_fuse else np.ones_like(net.W[l]))
                    net.W[l] += weight_update(M=M, theta=theta, Pi_post=net.Pi[l],
                                              eps_post=net.eps[l], elig=elig[l].value,
                                              eta=cfg.eta_w / cfg.tau_w)
                    net.B[l] += (1.0 / cfg.tau_b) * feedback_update(
                        net.B[l], a_up=net.x[l + 1], eps=net.eps[l], cfg=cfg)
                    net.Pi[l] = precision_update(net.Pi[l], eps_sq=net.eps[l] ** 2, cfg=cfg)

    A = tasks[0]
    last = tasks[-1]
    train(A)
    errA0 = _err_on(net, A, cfg, rng)
    errLast_before = _err_on(net, last, cfg, rng)
    forget = {}
    for m in range(1, n_tasks):
        train(tasks[m])
        forget[m] = _err_on(net, A, cfg, rng) - errA0
    errLast_after = _err_on(net, last, cfg, rng)
    cbar = float(np.mean([f.c.mean() for f in fuse])) if use_fuse else 0.0
    return {"forget_first": forget,
            "learn_last_drop": errLast_before - errLast_after,
            "errLast_after": errLast_after, "cbar": cbar}


def probe_forgetting_scaling(n_tasks=5, seeds=(0, 1, 2, 3, 4),
                             dim=10, per_task=6, passes=100):
    """Run the N-task continual stream with the fuse and with always-plastic learning.

    Returns out[method] = {
        'forget_first': {m: {mean,ci,raw}},   # forgetting of A after m more tasks
        'learn_last':   {mean,ci,raw},         # drop in last-task err after learning it
        'cbar':         float,                 # mean reserve (fuse only)
    }  for method in {'fuse','plastic'}.
    """
    out = {}
    for method, use_fuse in (("fuse", True), ("plastic", False)):
        per_seed = [_continual_n_tasks(use_fuse, n_tasks, seed=s, dim=dim,
                                       per_task=per_task, passes=passes) for s in seeds]
        forget = {}
        for m in range(1, n_tasks):
            forget[m] = _stat([r["forget_first"][m] for r in per_seed])
        out[method] = {
            "forget_first": forget,
            "learn_last": _stat([r["learn_last_drop"] for r in per_seed]),
            "cbar": float(np.mean([r["cbar"] for r in per_seed])),
        }
    return out


# ---------------------------------------------------------------------------
# (c) Deeper PC hierarchies on Task-1
# ---------------------------------------------------------------------------
def probe_depth_scaling(depths=(2, 3, 4), Ks=(5, 10, 20), seeds=(0, 1, 2, 3, 4),
                        h=6, w=6, vocab=8, n_settle=10):
    """Task-1 graph-completion with deeper CEREBRUM PC hierarchies.

    depth = number of PC AREAS (dims length). depth=2 -> (vocab, lat); depth=3 ->
    (vocab, lat, lat); etc. The grid HEAD/decode is unchanged — this isolates the
    effect of stacking more error-neuron areas on the sample-efficiency axis.

    Returns out[depth][K] = {'cerebrum': {mean,ci,raw}, 'chance': 1/vocab}.
    """
    out = {}
    lat = max(8, 2 * vocab)
    for depth in depths:
        out[depth] = {}
        dims = tuple([vocab] + [lat] * (depth - 1))
        for K in Ks:
            g = []
            for s in seeds:
                ep = make_episode(h=h, w=w, vocab=vocab, K=K, seed=s)
                cfg = CerebrumConfig(dims=dims, grid_n_modules=8, n_settle=n_settle, seed=s)
                g.append(run_cerebrum_episode(CerebrumCore(cfg), ep))
            out[depth][K] = {"cerebrum": _stat(g), "chance": 1.0 / vocab}
    return out


# ---------------------------------------------------------------------------
# Honest verdict helpers
# ---------------------------------------------------------------------------
def verdict_task1(res):
    """HONEST per-size verdict: does CEREBRUM-grid's advantage over the better of
    {flat, backprop-MLP} hold (CI-separated), shrink, or break as the graph grows?"""
    lines = []
    sizes = sorted(res, key=lambda s: (s[0] * s[1], s[2]))
    for size in sizes:
        h, w, vocab = size
        margins = []  # over K: (cerebrum_mean - best_baseline_mean), CI-separation flag
        for K in sorted(res[size]):
            row = res[size][K]
            gm, gci = row["cerebrum"]["mean"], row["cerebrum"]["ci"]
            bm = max(row["flat"]["mean"], row["mlp"]["mean"])
            bci = (row["flat"]["ci"] if row["flat"]["mean"] >= row["mlp"]["mean"]
                   else row["mlp"]["ci"])
            sep = (gm - gci) > (bm + bci)        # CEREBRUM CI strictly above best baseline CI
            margins.append((K, gm - bm, sep))
        any_win = any(sep for _, _, sep in margins)
        all_win = all(sep for _, _, sep in margins)
        first_margin = margins[0][1]
        last_margin = margins[-1][1]
        if all_win:
            tag = f"HOLDS (grid advantage CI-separated at every K; margin {first_margin:+.2f}->{last_margin:+.2f})"
        elif any_win:
            tag = (f"PARTIAL (grid advantage CI-separated at some K only; "
                   f"margin {first_margin:+.2f}->{last_margin:+.2f})")
        else:
            tag = (f"BREAKS (no CI-separated grid advantage at this size; "
                   f"margin {first_margin:+.2f}->{last_margin:+.2f})")
        lines.append(f"  {h}x{w} vocab={vocab}: {tag}")
    return "Task-1 grid-prior advantage vs best baseline:\n" + "\n".join(lines)


def verdict_forgetting(res):
    """HONEST verdict: does the fuse still reduce forgetting of the FIRST task,
    relative to always-plastic, as MORE tasks arrive? Report per-step and the
    final step with CI separation."""
    fuse = res["fuse"]["forget_first"]
    plastic = res["plastic"]["forget_first"]
    ms = sorted(fuse)
    lines = []
    holds_through = 0
    broke_at = None
    for m in ms:
        fm, fci = fuse[m]["mean"], fuse[m]["ci"]
        pm, pci = plastic[m]["mean"], plastic[m]["ci"]
        lower = fm < pm
        sep = (fm + fci) < (pm - pci)            # fuse CI strictly below plastic CI
        flag = "CI-sep" if sep else ("lower" if lower else "NOT lower")
        lines.append(f"  after +{m} tasks: fuse forget {fm:+.3f}+/-{fci:.3f} vs "
                     f"plastic {pm:+.3f}+/-{pci:.3f}  [{flag}]")
        if lower:
            holds_through = m
        elif broke_at is None:
            broke_at = m
    cbar = res["fuse"]["cbar"]
    if broke_at is None:
        head = (f"Fuse reduces first-task forgetting through ALL {ms[-1]} extra tasks "
                f"(cbar={cbar:.2f}).")
    else:
        head = (f"Fuse reduces first-task forgetting through +{holds_through} extra tasks; "
                f"advantage no longer holds at +{broke_at} (cbar={cbar:.2f}).")
    return "Task-2 fuse forgetting-protection as tasks accumulate:\n" + head + "\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI report
# ---------------------------------------------------------------------------
def _fmt(stat):
    return f"{stat['mean']:.3f} +/- {stat['ci']:.3f}"


def main():
    print("=" * 78)
    print("CEREBRUM SCALING PROBE — honest, exploratory (UNPROVEN BET; spec §7 OP#1)")
    print("=" * 78)

    # (a) Task-1 at larger sizes. 8 seeds: at n=5 the CI half-widths (esp. at K=5 on
    # small grids with few held-out queries) are wide enough to mask a large MEAN
    # advantage; 8 seeds tighten them so the HOLDS/BREAKS verdict tracks the means.
    print("\n(a) Task-1 few-shot graph-completion at larger sizes")
    print("    (mean +/- 95% CI over 8 seeds; CEREBRUM-grid vs flat-prior vs backprop-MLP)")
    t1 = probe_task1_scaling(seeds=(0, 1, 2, 3, 4, 5, 6, 7))
    for size in sorted(t1, key=lambda s: (s[0] * s[1], s[2])):
        h, w, vocab = size
        print(f"\n  gridworld {h}x{w}, vocab={vocab} (chance={1.0/vocab:.3f})")
        print(f"    {'K':>4}  {'CEREBRUM-grid':>18}  {'flat-prior':>18}  {'backprop-MLP':>18}")
        for K in sorted(t1[size]):
            row = t1[size][K]
            print(f"    {K:>4}  {_fmt(row['cerebrum']):>18}  {_fmt(row['flat']):>18}  {_fmt(row['mlp']):>18}")
    print()
    print(verdict_task1(t1))

    # (b) Forgetting with more tasks
    print("\n(b) Catastrophic forgetting with MORE sequential tasks (A->B->C->D->E)")
    fr = probe_forgetting_scaling(seeds=(0, 1, 2, 3, 4, 5, 6, 7))
    print("    forgetting of FIRST task (A) after each further task (lower=better):")
    ms = sorted(fr["fuse"]["forget_first"])
    print(f"    {'method':<10}" + "".join(f"{'+'+str(m):>16}" for m in ms))
    for method in ("fuse", "plastic"):
        cells = "".join(f"{_fmt(fr[method]['forget_first'][m]):>16}" for m in ms)
        print(f"    {method:<10}{cells}")
    print(f"    learn_last drop: fuse {_fmt(fr['fuse']['learn_last'])}, "
          f"plastic {_fmt(fr['plastic']['learn_last'])}")
    print()
    print(verdict_forgetting(fr))

    # (c) Deeper hierarchies
    print("\n(c) Deeper PC hierarchies on Task-1 (6x6, vocab=8)")
    dr = probe_depth_scaling(seeds=(0, 1, 2, 3, 4, 5, 6, 7))
    print(f"    {'depth(areas)':<14}" + "".join(f"{'K='+str(K):>16}" for K in (5, 10, 20)))
    for depth in sorted(dr):
        cells = "".join(f"{_fmt(dr[depth][K]['cerebrum']):>16}" for K in sorted(dr[depth]))
        print(f"    {depth:<14}{cells}")
    print("    (Task-1 completion is grid-HEAD driven; extra PC areas add error-neuron")
    print("     depth but do not change the path-integration mechanism — expect flat.)")

    print("\n" + "=" * 78)
    print("HONEST VERDICT: this is an UNPROVEN scaling bet, not 'scaling solved'.")
    print("See the per-axis tags above for exactly where advantages hold and break.")
    print("=" * 78)


if __name__ == "__main__":
    main()
