import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
import numpy as np
from cerebrum.config import CerebrumConfig
from cerebrum.network import CerebrumCore
from benchmarks.tasks.transitive import make_episode
from benchmarks.tasks.transitive_cerebrum import run_cerebrum_episode
from benchmarks.baselines.flat_prior_transitive import run_flat_episode
from benchmarks.baselines.backprop_mlp_transitive import run_mlp_episode


def run_sweep(exposures_list=(1, 2, 4), seeds=(0, 1, 2, 3, 4), n_items=7, vocab=10):
    """Sweep over #adjacent-pair EXPOSURES (the few-shot axis), at fixed order length.

    out[key][E] = mean over seeds (float); out[key+'_raw'][E] = per-seed list (for CIs).
    """
    out = {"cerebrum": {}, "flat": {}, "mlp": {},
           "cerebrum_raw": {}, "flat_raw": {}, "mlp_raw": {}}
    for E in exposures_list:
        g, f, m = [], [], []
        for s in seeds:
            ep = make_episode(n_items=n_items, vocab=vocab, exposures=E, seed=s)
            cfg = CerebrumConfig(dims=(vocab, 8, 8), grid_n_modules=8, n_settle=10, seed=s)
            g.append(run_cerebrum_episode(CerebrumCore(cfg), ep))
            f.append(run_flat_episode(ep))
            m.append(run_mlp_episode(ep, epochs=300, seed=s))
        out["cerebrum"][E] = float(np.mean(g)); out["cerebrum_raw"][E] = g
        out["flat"][E] = float(np.mean(f)); out["flat_raw"][E] = f
        out["mlp"][E] = float(np.mean(m)); out["mlp_raw"][E] = m
    return out


def run_length_sweep(n_items_list=(7, 15, 25), seeds=(0, 1, 2, 3, 4), exposures=1):
    """Sweep over ORDER LENGTH N at a single adjacent exposure (the discriminating axis:
    longer chains require chaining more transitive steps from the same few-shot budget)."""
    out = {"cerebrum": {}, "flat": {}, "mlp": {},
           "cerebrum_raw": {}, "flat_raw": {}, "mlp_raw": {}}
    for N in n_items_list:
        vocab = max(N + 3, 10)
        g, f, m = [], [], []
        for s in seeds:
            ep = make_episode(n_items=N, vocab=vocab, exposures=exposures, seed=s)
            cfg = CerebrumConfig(dims=(vocab, 8, 8), grid_n_modules=8, n_settle=10, seed=s)
            g.append(run_cerebrum_episode(CerebrumCore(cfg), ep))
            f.append(run_flat_episode(ep))
            m.append(run_mlp_episode(ep, epochs=400, seed=s))
        out["cerebrum"][N] = float(np.mean(g)); out["cerebrum_raw"][N] = g
        out["flat"][N] = float(np.mean(f)); out["flat_raw"][N] = f
        out["mlp"][N] = float(np.mean(m)); out["mlp_raw"][N] = m
    return out


if __name__ == "__main__":
    from benchmarks.stats import fmt_ci
    print("Transitive-inference few-shot accuracy on held-out NON-adjacent pairs")
    print("(mean +/- 95% CI over 5 seeds; chance = 0.500)\n")

    res = run_sweep()
    print("[A] vs adjacent-pair EXPOSURES  (N=7 items, vocab=10)")
    print(f"{'exposures':>9}  {'CEREBRUM-grid':>18}  {'flat-prior':>18}  {'backprop-MLP':>18}")
    for E in sorted(res["cerebrum"]):
        print(f"{E:>9}  {fmt_ci(res['cerebrum_raw'][E]):>18}  "
              f"{fmt_ci(res['flat_raw'][E]):>18}  {fmt_ci(res['mlp_raw'][E]):>18}")

    resN = run_length_sweep()
    print("\n[B] vs ORDER LENGTH N  (1 exposure each adjacent pair; the discriminating axis)")
    print(f"{'N_items':>9}  {'CEREBRUM-grid':>18}  {'flat-prior':>18}  {'backprop-MLP':>18}")
    for N in sorted(resN["cerebrum"]):
        print(f"{N:>9}  {fmt_ci(resN['cerebrum_raw'][N]):>18}  "
              f"{fmt_ci(resN['flat_raw'][N]):>18}  {fmt_ci(resN['mlp_raw'][N]):>18}")
