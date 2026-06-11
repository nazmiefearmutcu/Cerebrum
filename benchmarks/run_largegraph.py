"""Large-graph scaling probe for Task-1 few-shot graph-completion (HONEST).

This pushes the metric few-shot graph-completion task PAST the existing 8x8 ceiling
(see benchmarks/run_scaling.py, which stopped at 8x8 vocab=10) to 12x12 and 16x16
gridworlds with proportionally larger vocab, and asks ONE honest question:

    Does CEREBRUM's grid-prior sample-efficiency advantage over flat-prior and
    backprop-MLP HOLD, SHRINK, or BREAK as the graph grows?

The mechanistic tension we are testing:
  - CEREBRUM completes HELD-OUT edges by PATH INTEGRATION in a frozen grid code
    (start + displacement -> target grid code -> content store -> obs). This is a
    structural prior: it does NOT need to have observed the (start,target) edge.
  - flat-prior has random per-cell codes and NO transition algebra, so it can only
    recall AT bound codes -> near-chance on held-out queries.
  - backprop-MLP is the labeled comparator: it is TRAINED only on the K walked
    edges (start-onehot + displacement -> next obs), so at fixed small K on a big
    graph it sees a tiny, sparse supervision set.

  At a FIXED budget K, a bigger graph means each cell gets a SMALLER observed
  fraction (coverage = |observed cells| / (h*w)). CEREBRUM's ABSOLUTE accuracy may
  therefore fall as the graph grows (fewer cells have content bound, so fewer
  held-out targets are even decodable). That is expected and we report it honestly.
  The MEANINGFUL question is the MARGIN over the baselines under the SAME budget.

BAN COMPLIANCE: CEREBRUM path routes only through cerebrum/ primitives; the grid is
driven by Exogenous moves (graph_completion.run_cerebrum_episode); the only backprop
is the pre-existing benchmarks/baselines/backprop_mlp.py comparator. We do NOT
modify cerebrum/. The grid head keeps grid_n_modules=8 (same prior as the 8x8 probe);
its largest module period (~47 cells) still spans a 16x16 grid, so this is the
SAME structural prior, just a bigger graph — not a retuned advantage.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

import numpy as np

from cerebrum.config import CerebrumConfig
from cerebrum.network import CerebrumCore

from benchmarks.stats import mean_ci, fmt_ci
from benchmarks.tasks.gridworld import make_episode
from benchmarks.tasks.graph_completion import run_cerebrum_episode
from benchmarks.baselines.flat_prior import run_flat_episode
from benchmarks.baselines.backprop_mlp import run_mlp_episode


# Sizes to probe: (h, w, vocab). Vocab scales ~ with grid edge so chance falls and
# the few-shot decode stays non-trivial. 12x12 -> v12, 16x16 -> v16. We also keep
# 8x8 vocab=10 as the published reference point so the report is anchored.
DEFAULT_SIZES = ((8, 8, 10), (12, 12, 12), (16, 16, 16))
DEFAULT_KS = (10, 20, 40)
DEFAULT_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)


def _stat(raw):
    """Pack a per-seed list into {'mean','ci','raw'} (95% CI half-width over seeds)."""
    m, h = mean_ci(raw)
    return {"mean": float(m), "ci": float(h), "raw": [float(x) for x in raw]}


def probe_largegraph(sizes=DEFAULT_SIZES, Ks=DEFAULT_KS, seeds=DEFAULT_SEEDS,
                     n_settle=10, mlp_epochs=120):
    """For each (h,w,vocab) size and each K, score CEREBRUM-grid vs flat vs backprop-MLP.

    Returns out[(h,w,vocab)][K] = {
        cerebrum|flat|mlp: {mean,ci,raw},
        'chance':   1/vocab,
        'coverage': mean over seeds of |observed cells| / (h*w),  # how much of the
                    #   graph the K-step walk actually visits (fixed budget -> shrinks
                    #   as the graph grows; load-bearing for the honest verdict)
        'n_queries': mean over seeds of number of held-out queries scored,
    }
    The K observations are held FIXED across the three methods (same episode).
    """
    out = {}
    for (h, w, vocab) in sizes:
        out[(h, w, vocab)] = {}
        for K in Ks:
            g, f, m = [], [], []
            covs, nqs = [], []
            for s in seeds:
                ep = make_episode(h=h, w=w, vocab=vocab, K=K, seed=s)
                covs.append(len(ep.observed_cells) / float(h * w))
                nqs.append(len(ep.queries))
                # latent areas scale modestly with vocab so the decode has capacity;
                # grid head identical to the 8x8 probe (same structural prior).
                lat = max(8, 2 * vocab)
                cfg = CerebrumConfig(dims=(vocab, lat, lat), grid_n_modules=8,
                                  n_settle=n_settle, seed=s)
                g.append(run_cerebrum_episode(CerebrumCore(cfg), ep))
                f.append(run_flat_episode(ep))
                m.append(run_mlp_episode(ep, epochs=mlp_epochs, seed=s))
            out[(h, w, vocab)][K] = {
                "cerebrum": _stat(g), "flat": _stat(f), "mlp": _stat(m),
                "chance": 1.0 / vocab,
                "coverage": float(np.mean(covs)),
                "n_queries": float(np.mean(nqs)),
            }
    return out


def verdict_largegraph(res):
    """HONEST per-size verdict: does CEREBRUM-grid's advantage over the BETTER of
    {flat, backprop-MLP} HOLD (CI-separated at every K), SHRINK (CI-separated at
    some K only), or BREAK (no CI-separated advantage) as the graph grows?

    We report the MARGIN (cerebrum_mean - best_baseline_mean) at the first and last K,
    and we explicitly surface the coverage fraction so an absolute-accuracy drop is
    not mistaken for an advantage loss."""
    lines = []
    sizes = sorted(res, key=lambda s: (s[0] * s[1], s[2]))
    for size in sizes:
        h, w, vocab = size
        margins = []  # (K, cerebrum_mean - best_baseline_mean, ci_separated_flag)
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
        cov = res[size][sorted(res[size])[0]]["coverage"]  # coverage ~ same per size at fixed K0
        if all_win:
            tag = (f"HOLDS (grid advantage CI-separated at every K; "
                   f"margin {first_margin:+.2f}->{last_margin:+.2f})")
        elif any_win:
            tag = (f"SHRINKS (grid advantage CI-separated at some K only; "
                   f"margin {first_margin:+.2f}->{last_margin:+.2f})")
        else:
            tag = (f"BREAKS (no CI-separated grid advantage at this size; "
                   f"margin {first_margin:+.2f}->{last_margin:+.2f})")
        lines.append(f"  {h}x{w} vocab={vocab} (coverage~{cov:.2f} of cells): {tag}")
    return "Task-1 LARGE-GRAPH grid-prior advantage vs best baseline:\n" + "\n".join(lines)


def _fmt(stat):
    return f"{stat['mean']:.3f} +/- {stat['ci']:.3f}"


def main():
    sizes = DEFAULT_SIZES
    Ks = DEFAULT_KS
    seeds = DEFAULT_SEEDS
    print("=" * 86)
    print("CEREBRUM LARGE-GRAPH SCALING PROBE — Task-1 few-shot graph-completion (HONEST)")
    print("Pushing past 8x8 to 12x12 and 16x16 with proportionally larger vocab.")
    print("=" * 86)
    print(f"\n  seeds={list(seeds)}  Ks={list(Ks)}  (mean +/- 95% CI over {len(seeds)} seeds)")
    print("  CEREBRUM-grid vs flat-prior vs backprop-MLP; chance = 1/vocab per row.\n")

    res = probe_largegraph(sizes=sizes, Ks=Ks, seeds=seeds)

    for size in sorted(res, key=lambda s: (s[0] * s[1], s[2])):
        h, w, vocab = size
        cov0 = res[size][sorted(res[size])[0]]["coverage"]
        print(f"  gridworld {h}x{w}, vocab={vocab}  (chance={1.0/vocab:.3f}, "
              f"cells={h*w}, coverage at K={sorted(Ks)[0]}~{cov0:.2f})")
        print(f"    {'K':>4}  {'coverage':>9}  {'#queries':>9}  "
              f"{'CEREBRUM-grid':>18}  {'flat-prior':>18}  {'backprop-MLP':>18}")
        for K in sorted(res[size]):
            row = res[size][K]
            print(f"    {K:>4}  {row['coverage']:>9.2f}  {row['n_queries']:>9.1f}  "
                  f"{_fmt(row['cerebrum']):>18}  {_fmt(row['flat']):>18}  {_fmt(row['mlp']):>18}")
        print()

    print(verdict_largegraph(res))

    print("\n" + "=" * 86)
    print("HONEST NOTE: at FIXED K a bigger graph = SMALLER observed coverage fraction,")
    print("so CEREBRUM's ABSOLUTE completion accuracy is expected to fall as h*w grows.")
    print("The verdict above is about the MARGIN over flat/backprop, not absolute level.")
    print("=" * 86)


if __name__ == "__main__":
    main()
