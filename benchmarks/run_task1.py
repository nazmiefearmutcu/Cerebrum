import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
import numpy as np
from grail.config import GRAILConfig
from grail.network import GRAILCore
from benchmarks.tasks.gridworld import make_episode
from benchmarks.tasks.graph_completion import run_grail_episode
from benchmarks.baselines.flat_prior import run_flat_episode
from benchmarks.baselines.backprop_mlp import run_mlp_episode

def run_sweep(Ks=(5,10,20), seeds=(0,1,2), h=4, w=4, vocab=5):
    out = {"grail":{}, "flat":{}, "mlp":{}}
    for K in Ks:
        g=[]; f=[]; m=[]
        for s in seeds:
            ep = make_episode(h=h, w=w, vocab=vocab, K=K, seed=s)
            cfg = GRAILConfig(dims=(vocab,8,8), grid_n_modules=8, n_settle=10, seed=s)
            g.append(run_grail_episode(GRAILCore(cfg), ep))
            f.append(run_flat_episode(ep))
            m.append(run_mlp_episode(ep, epochs=80))
        out["grail"][K]=float(np.mean(g)); out["flat"][K]=float(np.mean(f)); out["mlp"][K]=float(np.mean(m))
    return out

if __name__ == "__main__":
    res = run_sweep()
    print(f"{'K':>4} {'GRAIL-grid':>12} {'flat-prior':>12} {'backprop-MLP':>14}")
    for K in sorted(res['grail']):
        print(f"{K:>4} {res['grail'][K]:>12.3f} {res['flat'][K]:>12.3f} {res['mlp'][K]:>14.3f}")
