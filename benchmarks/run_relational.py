"""Run the NON-METRIC relational benchmark (FM7): CEREBRUM-grid vs flat-prior vs backprop-MLP.

This is the adversarial counterpart to run_task1.py. Same comparators, same CI machinery,
but on a random DIRECTED graph with asymmetric, non-commuting relation composition — a space
with NO metric embedding for the grid to path-integrate over. The expected and honest result
is that CEREBRUM's grid advantage (which is real on the metric gridworld of Task-1) does NOT
carry over here.

We run this on two tasks:
1. The original Random Directed Graph task.
2. The new Directed Tree/Hierarchy Graph task.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
import numpy as np
from cerebrum.config import CerebrumConfig
from cerebrum.core_net import CerebrumCore
from benchmarks.tasks.relational import make_episode, run_cerebrum_episode, RelationalGraph, TreeRelationalGraph
from benchmarks.baselines.flat_prior_relational import run_flat_relational_episode
from benchmarks.baselines.backprop_mlp_relational import run_mlp_relational_episode


def run_sweep(Ks=(5, 10, 20), seeds=(0, 1, 2, 3, 4), n_nodes=16, n_relations=3, vocab=5, graph_class=RelationalGraph):
    out = {"cerebrum": {}, "flat": {}, "mlp": {}, "cerebrum_raw": {}, "flat_raw": {}, "mlp_raw": {}}
    for K in Ks:
        g = []; f = []; m = []
        for s in seeds:
            ep = make_episode(
                n_nodes=n_nodes, 
                n_relations=n_relations, 
                vocab=vocab, 
                K=K, 
                seed=s, 
                graph_class=graph_class
            )
            cfg = CerebrumConfig(dims=(vocab, 8, 8), grid_n_modules=8, n_settle=10, seed=s)
            g.append(run_cerebrum_episode(CerebrumCore(cfg), ep))
            f.append(run_flat_relational_episode(ep))
            m.append(run_mlp_relational_episode(ep, epochs=80))
        out["cerebrum"][K] = float(np.mean(g)); out["flat"][K] = float(np.mean(f)); out["mlp"][K] = float(np.mean(m))
        out["cerebrum_raw"][K] = g; out["flat_raw"][K] = f; out["mlp_raw"][K] = m
    return out


if __name__ == "__main__":
    from benchmarks.stats import fmt_ci
    vocab = 5
    
    # 1. Sweep over Random Directed Graph Task
    print("=" * 80)
    print("SWEEP 1: Random Directed Graph (Asymmetric, non-metric, non-commuting)")
    res_rand = run_sweep(vocab=vocab, graph_class=RelationalGraph)
    print(f"FM7 NON-METRIC relational completion (mean +/- 95% CI over 5 seeds; chance=1/vocab={1.0/vocab:.3f})")
    print("Directed graph, asymmetric & non-commuting relations -> no metric embedding for the grid.")
    print(f"{'K':>4}  {'CEREBRUM-grid':>18}  {'flat-prior':>18}  {'backprop-MLP':>18}")
    for K in sorted(res_rand['cerebrum']):
        print(f"{K:>4}  {fmt_ci(res_rand['cerebrum_raw'][K]):>18}  {fmt_ci(res_rand['flat_raw'][K]):>18}  {fmt_ci(res_rand['mlp_raw'][K]):>18}")
    
    # 2. Sweep over Directed Tree/Hierarchy Graph Task
    print("\n" + "=" * 80)
    print("SWEEP 2: Directed Tree/Hierarchy Graph (Asymmetric, non-metric, non-commuting)")
    res_tree = run_sweep(vocab=vocab, graph_class=TreeRelationalGraph)
    print(f"FM7 TREE-HIERARCHY relational completion (mean +/- 95% CI over 5 seeds; chance=1/vocab={1.0/vocab:.3f})")
    print("Directed tree, asymmetric & non-commuting child/parent relations -> no metric embedding for the grid.")
    print(f"{'K':>4}  {'CEREBRUM-grid':>18}  {'flat-prior':>18}  {'backprop-MLP':>18}")
    for K in sorted(res_tree['cerebrum']):
        print(f"{K:>4}  {fmt_ci(res_tree['cerebrum_raw'][K]):>18}  {fmt_ci(res_tree['flat_raw'][K]):>18}  {fmt_ci(res_tree['mlp_raw'][K]):>18}")
    print("=" * 80)
