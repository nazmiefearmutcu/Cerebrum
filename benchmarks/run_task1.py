import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
import numpy as np
from cerebrum.config import CerebrumConfig
from cerebrum.core_net import CerebrumCore
from benchmarks.tasks.gridworld import make_episode
from benchmarks.tasks.graph_completion import run_cerebrum_episode
from benchmarks.baselines.flat_prior import run_flat_episode
from benchmarks.baselines.backprop_mlp import run_mlp_episode

# OPT-IN measurement toggle (default OFF -> behavior unchanged). When
# CEREBRUM_BALANCE_GRID_PRECISION=1 the CEREBRUM grid-completion model carries the new
# precision-balancing flag, so we can confirm the grid few-shot win still holds.
_BALANCE = os.environ.get("CEREBRUM_BALANCE_GRID_PRECISION", "0") not in ("0", "", "false", "False")

def run_sweep(Ks=(5,10,20), seeds=(0,1,2,3,4), h=4, w=4, vocab=5):
    # out["cerebrum"][K] = mean (backward-compatible float); out["cerebrum_raw"][K] = per-seed list (for CIs)
    out = {"cerebrum":{}, "flat":{}, "mlp":{}, "cerebrum_raw":{}, "flat_raw":{}, "mlp_raw":{}}
    for K in Ks:
        g=[]; f=[]; m=[]
        for s in seeds:
            ep = make_episode(h=h, w=w, vocab=vocab, K=K, seed=s)
            cfg = CerebrumConfig(dims=(vocab,8,8), grid_n_modules=8, n_settle=10, seed=s,
                              balance_grid_precision=_BALANCE)
            g.append(run_cerebrum_episode(CerebrumCore(cfg), ep))
            f.append(run_flat_episode(ep))
            m.append(run_mlp_episode(ep, epochs=80))
        out["cerebrum"][K]=float(np.mean(g)); out["flat"][K]=float(np.mean(f)); out["mlp"][K]=float(np.mean(m))
        out["cerebrum_raw"][K]=g; out["flat_raw"][K]=f; out["mlp_raw"][K]=m
    return out

if __name__ == "__main__":
    from benchmarks.stats import fmt_ci
    res = run_sweep()
    print("Task-1 few-shot graph-completion (mean +/- 95% CI over 5 seeds; chance=1/vocab=0.200)")
    print(f"{'K':>4}  {'CEREBRUM-grid':>18}  {'flat-prior':>18}  {'backprop-MLP':>18}")
    for K in sorted(res['cerebrum']):
        print(f"{K:>4}  {fmt_ci(res['cerebrum_raw'][K]):>18}  {fmt_ci(res['flat_raw'][K]):>18}  {fmt_ci(res['mlp_raw'][K]):>18}")
