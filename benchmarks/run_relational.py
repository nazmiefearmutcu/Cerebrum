"""Run the NON-METRIC relational benchmark (FM7): GRAIL-grid vs flat-prior vs backprop-MLP.

This is the adversarial counterpart to run_task1.py. Same comparators, same CI machinery,
but on a random DIRECTED graph with asymmetric, non-commuting relation composition — a space
with NO metric embedding for the grid to path-integrate over. The expected and honest result
is that GRAIL's grid advantage (which is real on the metric gridworld of Task-1) does NOT
carry over here.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
import numpy as np
from grail.config import GRAILConfig
from grail.network import GRAILCore
from benchmarks.tasks.relational import make_episode, make_hierarchy_episode, run_grail_episode
from benchmarks.baselines.flat_prior_relational import run_flat_relational_episode
from benchmarks.baselines.backprop_mlp_relational import run_mlp_relational_episode


def run_sweep(Ks=(5, 10, 20), seeds=(0, 1, 2, 3, 4), n_nodes=16, n_relations=3, vocab=5, use_hierarchy=False):
    out = {"grail": {}, "flat": {}, "mlp": {}, "grail_raw": {}, "flat_raw": {}, "mlp_raw": {}}
    for K in Ks:
        g = []; f = []; m = []
        for s in seeds:
            if use_hierarchy:
                ep = make_hierarchy_episode(n_nodes=n_nodes, vocab=vocab, K=K, seed=s)
            else:
                ep = make_episode(n_nodes=n_nodes, n_relations=n_relations, vocab=vocab, K=K, seed=s)
            cfg = GRAILConfig(dims=(vocab, 8, 8), grid_n_modules=8, n_settle=10, seed=s)
            g.append(run_grail_episode(GRAILCore(cfg), ep))
            f.append(run_flat_relational_episode(ep))
            m.append(run_mlp_relational_episode(ep, epochs=80))
        out["grail"][K] = float(np.mean(g)); out["flat"][K] = float(np.mean(f)); out["mlp"][K] = float(np.mean(m))
        out["grail_raw"][K] = g; out["flat_raw"][K] = f; out["mlp_raw"][K] = m
    return out


if __name__ == "__main__":
    from benchmarks.stats import fmt_ci
    vocab = 5
    
    # 1. Original Random Directed Graph Sweep
    res_orig = run_sweep(vocab=vocab, use_hierarchy=False)
    print(f"FM7 NON-METRIC relational completion (mean +/- 95% CI over 5 seeds; chance=1/vocab={1.0/vocab:.3f})")
    print("Directed graph, asymmetric & non-commuting relations -> no metric embedding for the grid.")
    print(f"{'K':>4}  {'GRAIL-grid':>18}  {'flat-prior':>18}  {'backprop-MLP':>18}")
    for K in sorted(res_orig['grail']):
        print(f"{K:>4}  {fmt_ci(res_orig['grail_raw'][K]):>18}  {fmt_ci(res_orig['flat_raw'][K]):>18}  {fmt_ci(res_orig['mlp_raw'][K]):>18}")
    print()

    # 2. New Directed Tree/Hierarchy Graph Sweep
    res_hier = run_sweep(vocab=vocab, use_hierarchy=True)
    print(f"DIRECTED TREE/HIERARCHY relational completion (mean +/- 95% CI over 5 seeds; chance=1/vocab={1.0/vocab:.3f})")
    print("Heap-like binary tree indexing hierarchy graph.")
    print(f"{'K':>4}  {'GRAIL-grid':>18}  {'flat-prior':>18}  {'backprop-MLP':>18}")
    for K in sorted(res_hier['grail']):
        print(f"{K:>4}  {fmt_ci(res_hier['grail_raw'][K]):>18}  {fmt_ci(res_hier['flat_raw'][K]):>18}  {fmt_ci(res_hier['mlp_raw'][K]):>18}")

