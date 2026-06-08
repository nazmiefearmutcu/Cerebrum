"""COMPOSITIONAL-GENERALIZATION depth probe: does adding PC areas (depth 2 vs 3 vs 4) help
compositional generalization when the readout consults the DEEP PC HIERARCHY (not the grid HEAD)?

This turns the earlier I6 "depth doesn't help" NULL into a REAL probe. There, Task-1 completion
read only the grid-HEAD content store, so extra PC areas were never on the causal path and the
null was uninformative. Here there is NO grid head: the bare PC hierarchy must complete a masked
input by pattern-completion, so the number of PC areas is genuinely consulted.

We sweep dims = (obs,h) [2 areas] vs (obs,h,h) [3] vs (obs,h,h,h) [4] at FIXED width and FIXED
training budget, over >=5 seeds, and report compositional-completion accuracy on held-out combos
(mean +/- 95% CI). We also report train-combo completion (a within-distribution reference) and
two baselines: a flat memorizer (the 'cannot compose' floor) and a backprop-MLP comparator.

The VERDICT (depth helps / helps-a-bit / null) is printed from the actual numbers — nothing is
engineered to make depth win.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
import numpy as np
from grail.config import GRAILConfig
from benchmarks.stats import mean_ci, fmt_ci
from benchmarks.tasks.compositional import (
    CompositionalTask, run_pc_completion, run_flat_memorizer, run_backprop_mlp,
)


def run_depth_sweep(depths=(2, 3, 4), width=24, A=4, B=4, part_dim=8,
                    passes=80, seeds=(0, 1, 2, 3, 4)):
    """For each PC depth, train+score over seeds. depth=2 -> dims=(obs,h); 3 -> (obs,h,h); etc.

    Returns per-depth raw per-seed lists for held-out and train completion accuracy, plus the
    seed-pooled flat-memorizer and backprop-MLP references (these do not depend on PC depth)."""
    # one task per seed (same tasks reused across depths so depth is the only thing that varies)
    out = {"heldout_raw": {}, "train_raw": {}, "lat_raw": {}, "compnorm_raw": {}}
    mem_raw, mlp_raw = [], []
    tasks = {s: CompositionalTask(A=A, B=B, part_dim=part_dim, seed=s) for s in seeds}
    for s in seeds:
        mem_raw.append(run_flat_memorizer(tasks[s]))
        mlp_raw.append(run_backprop_mlp(tasks[s], epochs=400, seed=s))
    for d in depths:
        held, trn, lat, cnorm = [], [], [], []
        for s in seeds:
            task = tasks[s]
            dims = tuple([task.obs_dim] + [width] * (d - 1))
            cfg = GRAILConfig(dims=dims, n_settle=12, seed=s)
            res = run_pc_completion(task, cfg, passes=passes)
            held.append(res["acc_heldout"]); trn.append(res["acc_train"])
            lat.append(res["lat_act"]); cnorm.append(res["comp_norm"])
        out["heldout_raw"][d] = held
        out["train_raw"][d] = trn
        out["lat_raw"][d] = lat
        out["compnorm_raw"][d] = cnorm
    out["mem_raw"] = mem_raw
    out["mlp_raw"] = mlp_raw
    out["meta"] = {"width": width, "A": A, "B": B, "part_dim": part_dim,
                   "passes": passes, "seeds": list(seeds),
                   "n_heldout": len(tasks[seeds[0]].heldout_combos),
                   "n_train": len(tasks[seeds[0]].train_combos),
                   "chance": 1.0 / B}
    return out


def _verdict(out, depths):
    """Decide depth helps / helps-a-bit / null from CI-separation of held-out accuracy."""
    shallow = out["heldout_raw"][depths[0]]
    deep = out["heldout_raw"][depths[-1]]
    ms, hs = mean_ci(shallow)
    md, hd = mean_ci(deep)
    diff = md - ms
    # CI-separated improvement: deeper lower bound above shallower upper bound
    if (md - hd) > (ms + hs) and diff > 0.05:
        return f"DEPTH HELPS: deepest ({depths[-1]}) beats shallowest ({depths[0]}) by {diff:+.3f} (95% CIs separated)."
    if diff > 0.03 and (md - hd) > ms:
        return f"DEPTH HELPS A BIT: deepest is {diff:+.3f} higher than shallowest, but CIs are not cleanly separated."
    if abs(diff) <= 0.03:
        return f"NULL: depth does not change compositional generalization (deepest - shallowest = {diff:+.3f}, within noise)."
    return (f"NULL / NEGATIVE: depth does not help (deepest - shallowest = {diff:+.3f}); "
            f"any movement is not a CI-separated improvement.")


if __name__ == "__main__":
    depths = (2, 3, 4)
    out = run_depth_sweep(depths=depths)
    m = out["meta"]
    print("COMPOSITIONAL-GENERALIZATION via DEEP PC-hierarchy pattern-completion (NO grid head)")
    print(f"A={m['A']} f1 values x B={m['B']} f2 values; part_dim={m['part_dim']} "
          f"(obs_dim={2*m['part_dim']}); width={m['width']}; passes={m['passes']}; "
          f"seeds={len(m['seeds'])}; chance=1/B={m['chance']:.3f}")
    print(f"train combos={m['n_train']}, held-out combos={m['n_heldout']} per seed")
    print()
    print(f"{'PC depth':>9}  {'held-out acc (95% CI)':>26}  {'train acc (95% CI)':>26}  "
          f"{'latent |x|':>11}  {'comp f2 norm':>12}")
    print(f"{'(areas)':>9}  {'[compositional]':>26}  {'[within-dist. ref]':>26}  "
          f"{'[diag]':>11}  {'[diag]':>12}")
    for d in depths:
        lat = float(np.mean(out['lat_raw'][d])); cn = float(np.mean(out['compnorm_raw'][d]))
        print(f"{d:>9}  {fmt_ci(out['heldout_raw'][d]):>26}  {fmt_ci(out['train_raw'][d]):>26}  "
              f"{lat:>11.4f}  {cn:>12.4f}")
    print()
    print(f"{'baseline':>22}  {'held-out acc (95% CI)':>26}")
    print(f"{'flat memorizer':>22}  {fmt_ci(out['mem_raw']):>26}   <- 'cannot compose' floor")
    print(f"{'backprop-MLP (comp.)':>22}  {fmt_ci(out['mlp_raw']):>26}   <- gradient comparator")
    print()
    print("VERDICT:", _verdict(out, depths))
