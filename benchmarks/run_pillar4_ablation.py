"""Pillar-4 ablation: does the stochastic settling noise (T_floor > 0) actually BUY anything?

GRAIL's Pillar 4 claims the Langevin settling noise term  noise = sqrt(2 tau_x T dt) dW  in
pc_core.settle_step is LOAD-BEARING: a noise floor T_floor > 0 "forbids MAP collapse" and is
"the brain's Monte Carlo". This script tests that claim HONESTLY by sweeping the SETTLING
temperature T_floor over a range that INCLUDES 0.0 (the deterministic / MAP limit) and measuring
the effect on all three core axes with >= 5 seeds and 95% CIs:

  (1) Task-1  : few-shot graph-completion accuracy.
  (2) Stage-2 : routing accuracy AND win-entropy / load-balance (dead-expert / hog collapse?).
  (3) Stage-3 : first-task forgetA (continual retention; lower is better).

T_floor is injected ONLY through the existing GRAILConfig(T_floor=...).  Every OTHER knob is held
at each task's existing working value (copied verbatim from benchmarks/run_task1.py,
benchmarks/tasks/binding.py defaults, and benchmarks/tasks/continual._make_cfg). The deterministic
arm is T_floor = 0.0.

WHERE THE NOISE ENTERS EACH AXIS (so the verdict has a mechanism):
  - settle_step adds  rng.normal(scale=sqrt(2 T dt / tau_x))  to every NON-clamped area's activity.
  - Task-1 reads out via grid.complete() = (Hebbian content store) @ (deterministic grid code).
    The PC settled activity x NEVER feeds back into that store or into predict_obs_here, so the
    completion readout is structurally INDEPENDENT of settling noise (expect an exact null).
  - Stage-2 gate bids on  pi * err_sq  where err_sq is each module's SETTLED error magnitude, so
    noise perturbs the bid -> can break ties / corrupt the salient module's lead.
  - Stage-3 trains with T=T_floor but MEASURES at T=0 (continual.py design), so forgetA reflects
    how noisy settling shaped the learned WEIGHTS, not eval-time jitter.

This file does NOT modify grail/. It re-implements each task's run loop locally so T_floor can be
threaded through cfg while keeping all other knobs identical to the shipped tasks.
Run:  python3 benchmarks/run_pillar4_ablation.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
from dataclasses import replace
import numpy as np

from grail.config import GRAILConfig
from grail.network import GRAILCore
from grail.network2 import GRAILWorkspaceNet
from grail.pc_core import PCAreas
from grail.plasticity import Eligibility, weight_update, precision_update, feedback_update
from grail.metaplasticity import MetaplasticFuse
from grail.neuromod import Neuromodulator
from grail.rng import SeededRNG

from benchmarks.tasks.gridworld import make_episode
from benchmarks.tasks.graph_completion import run_grail_episode
import benchmarks.tasks.continual as continual
from benchmarks.stats import mean_ci, fmt_ci

# The swept settling temperatures. 0.0 = deterministic / MAP limit; 0.02 = shipped default.
T_GRID = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)
SEEDS = (0, 1, 2, 3, 4)


# ----------------------------------------------------------------------------------------------
# Axis 1: Task-1 few-shot graph-completion. Knobs verbatim from benchmarks/run_task1.py.
# ----------------------------------------------------------------------------------------------
def task1_acc(T_floor, K, seed, h=4, w=4, vocab=5):
    ep = make_episode(h=h, w=w, vocab=vocab, K=K, seed=seed)
    cfg = GRAILConfig(dims=(vocab, 8, 8), grid_n_modules=8, n_settle=10, seed=seed, T_floor=T_floor)
    return run_grail_episode(GRAILCore(cfg), ep)


# ----------------------------------------------------------------------------------------------
# Axis 2: Stage-2 routing. Loop + knobs verbatim from benchmarks/tasks/binding.run_binding,
# with T_floor injected into the cfg (the only change).
# ----------------------------------------------------------------------------------------------
def stage2_routing(T_floor, n_modules, trials, seed,
                   explore_reward=2.0, reward_scale=5.0, gate_temp=0.1, lam_g=0.05):
    rng = np.random.default_rng(seed)
    cfg = GRAILConfig(dims=(n_modules, n_modules), n_settle=6, seed=seed,
                      lam_g=lam_g, gate_temp=gate_temp, T_floor=T_floor)
    net = GRAILWorkspaceNet(n_modules, 1, slice_dim=n_modules, cfg=cfg)
    wins = np.zeros(n_modules); correct = 0
    for _ in range(trials):
        target = int(rng.integers(0, n_modules))
        obs = [np.zeros(n_modules) for _ in range(n_modules)]
        for m in range(n_modules):
            obs[m][rng.integers(0, n_modules)] = 1.0
        obs[target][:] = 0.0; obs[target][target] = 2.0      # salient (rewarded) object
        z, _ = net.step(obs, reward=explore_reward)          # exploration pass (low T_gate)
        winner = int(np.argmax(z[:, 0]))
        reward = reward_scale if winner == target else 0.0
        z, _ = net.step(obs, reward=reward)                  # learning pass
        winner = int(np.argmax(z[:, 0])); wins[winner] += 1
        if winner == target:
            correct += 1
    p = wins / wins.sum()
    ent = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    # min module-win share: 0 => at least one DEAD expert; high => balanced load.
    min_share = float(wins.min() / wins.sum())
    # max module-win share: 1 => one module HOGS all slots.
    max_share = float(wins.max() / wins.sum())
    return {"routing_acc": correct / trials, "win_entropy": ent,
            "min_share": min_share, "max_share": max_share, "wins": wins.tolist()}


# ----------------------------------------------------------------------------------------------
# Axis 3: Stage-3 continual forgetting. Loop + knobs verbatim from
# benchmarks/tasks/continual.run_continual, with T_floor injected into the cfg.
# TRAINING settles at T=T_floor (the swept knob); the MEASUREMENT readout stays T=0 by the
# shipped design (continual._err_on), so forgetA reflects the learned weights.
# ----------------------------------------------------------------------------------------------
def stage3_continual(T_floor, use_fuse, seed, dim=10, per_task=6, passes=100):
    cfg = replace(continual._make_cfg(seed, dim), T_floor=T_floor)
    rng_proto = np.random.default_rng(seed + 5)
    A, B, Cc = continual._prototypes(rng_proto, 3, per_task, dim)
    net = PCAreas(cfg); nm = Neuromodulator(cfg); rng = SeededRNG(seed)
    elig = [Eligibility((cfg.dims[l + 1],), cfg) for l in range(net.L - 1)]
    fuse = [MetaplasticFuse(net.W[l].shape, cfg) for l in range(net.L - 1)] if use_fuse else None

    def train(patterns):
        for _ in range(passes):
            for p in patterns:
                for _ in range(cfg.n_settle):
                    net.settle_step(rng, T=cfg.T_floor, clamp_bottom=p)   # <-- swept T_floor
                    for l in range(net.L - 1):
                        elig[l].step(a_pre=net.x[l + 1])
                net.compute_errors(); M = nm.update(reward=1.0)
                for l in range(net.L - 1):
                    theta = fuse[l].update(net.Pi[l], net.eps[l], elig[l].value) \
                        if use_fuse else np.ones_like(net.W[l])
                    net.W[l] += weight_update(M=M, theta=theta, Pi_post=net.Pi[l],
                                              eps_post=net.eps[l], elig=elig[l].value,
                                              eta=cfg.eta_w / cfg.tau_w)
                    net.B[l] += (1.0 / cfg.tau_b) * feedback_update(net.B[l], a_up=net.x[l + 1],
                                                                    eps=net.eps[l], cfg=cfg)
                    net.Pi[l] = precision_update(net.Pi[l], eps_sq=net.eps[l] ** 2, cfg=cfg)

    train(A); errA_afterA = continual._err_on(net, A, cfg, rng)
    train(B); train(Cc)
    errA_afterC = continual._err_on(net, A, cfg, rng)
    return {"forgetA": errA_afterC - errA_afterA, "errA_afterA": errA_afterA,
            "errA_afterC": errA_afterC}


# ----------------------------------------------------------------------------------------------
# Sweep drivers + reporting.
# ----------------------------------------------------------------------------------------------
def _verdict(by_T, key, lower_is_better=False, rel_eps=0.02):
    """Compare each T_floor>0 arm against the deterministic T=0 baseline and label the BEST
    non-zero arm vs T=0 as HELPS / NEUTRAL / HURTS using non-overlapping 95% CIs."""
    det_mean, det_h = mean_ci([r[key] for r in by_T[0.0]])
    best_T, best_mean, best_h, best_better = None, None, None, None
    for T in T_GRID:
        if T == 0.0:
            continue
        m, h = mean_ci([r[key] for r in by_T[T]])
        better = (m < det_mean) if lower_is_better else (m > det_mean)
        if best_mean is None or better and (m < best_mean if lower_is_better else m > best_mean):
            best_T, best_mean, best_h, best_better = T, m, h, better
    # CI-separated improvement over deterministic?
    if lower_is_better:
        sep = (best_mean + best_h) < (det_mean - det_h)
    else:
        sep = (best_mean - best_h) > (det_mean + det_h)
    if best_better and sep:
        label = f"NOISE HELPS (best T={best_T}, CI-separated from T=0)"
    elif best_better and abs(best_mean - det_mean) > rel_eps * (abs(det_mean) + 1e-9):
        label = f"noise helps (best T={best_T}) but within CI overlap -> WEAK"
    else:
        label = "NEUTRAL (no T>0 beats T=0 beyond CI)"
    return det_mean, det_h, best_T, best_mean, best_h, label


def run_axis1():
    Ks = (5, 10, 20)
    by_T = {T: [] for T in T_GRID}        # flattened across K for verdict
    table = {}                            # (K) -> {T: raw list}
    for K in Ks:
        table[K] = {T: [] for T in T_GRID}
        for s in SEEDS:
            for T in T_GRID:
                a = task1_acc(T, K=K, seed=s)
                table[K][T].append(a); by_T[T].append({"acc": a})
    return Ks, table, by_T


def run_axis2():
    Ms = (4, 6)
    table = {}
    by_T = {T: [] for T in T_GRID}
    for nm in Ms:
        table[nm] = {T: [] for T in T_GRID}
        for s in SEEDS:
            for T in T_GRID:
                r = stage2_routing(T, n_modules=nm, trials=400, seed=s)
                table[nm][T].append(r); by_T[T].append(r)
    return Ms, table, by_T


def run_axis3():
    by_T_fuse = {T: [] for T in T_GRID}
    by_T_plastic = {T: [] for T in T_GRID}
    for s in SEEDS:
        for T in T_GRID:
            by_T_fuse[T].append(stage3_continual(T, use_fuse=True, seed=s))
            by_T_plastic[T].append(stage3_continual(T, use_fuse=False, seed=s))
    return by_T_fuse, by_T_plastic


def main():
    print("=" * 92)
    print("PILLAR-4 ABLATION  —  does the stochastic settling noise (T_floor>0) buy anything?")
    print(f"  sweep T_floor in {T_GRID};  {len(SEEDS)} seeds;  mean +/- 95% CI (Student-t)")
    print(f"  T_floor=0.0 = deterministic / MAP settling.  0.02 = shipped default.")
    print("=" * 92)

    # ---- Axis 1 -------------------------------------------------------------------------------
    print("\n[AXIS 1] Task-1 few-shot graph-completion accuracy  (chance = 1/vocab = 0.200)")
    Ks, t1, by_T1 = run_axis1()
    header = "  K   " + "".join(f"{('T='+str(T)):>16}" for T in T_GRID)
    print(header)
    for K in Ks:
        row = f"{K:>4}  " + "".join(f"{fmt_ci(t1[K][T]):>16}" for T in T_GRID)
        print(row)
    dm, dh, bT, bm, bh, lab = _verdict(by_T1, "acc", lower_is_better=False)
    # Exactness check: is acc bit-identical across all T (per seed)?
    exact = all(len(set(round(t1[K][T][i], 12) for T in T_GRID)) == 1
                for K in Ks for i in range(len(SEEDS)))
    print(f"  T=0 baseline acc = {dm:.3f} +/- {dh:.3f};  best T>0 = {bT} -> {bm:.3f} +/- {bh:.3f}")
    print(f"  bit-exact across ALL T_floor (per K,seed): {exact}")
    print(f"  VERDICT axis-1: {lab}"
          + ("  [structural null: readout bypasses PC settling]" if exact else ""))

    # ---- Axis 2 -------------------------------------------------------------------------------
    print("\n[AXIS 2] Stage-2 routing accuracy + load balance  (higher acc better; higher entropy = balanced)")
    Ms, t2, by_T2 = run_axis2()
    for nm in Ms:
        print(f"  -- M={nm} modules (chance={1.0/nm:.3f}; max win_entropy=ln M={np.log(nm):.3f}) --")
        print("        metric  " + "".join(f"{('T='+str(T)):>16}" for T in T_GRID))
        for metric, prec in (("routing_acc", 3), ("win_entropy", 3),
                             ("min_share", 3), ("max_share", 3)):
            cells = "".join(f"{fmt_ci([r[metric] for r in t2[nm][T]], prec):>16}" for T in T_GRID)
            print(f"  {metric:>12}  {cells}")
    dm, dh, bT, bm, bh, lab = _verdict(by_T2, "routing_acc", lower_is_better=False)
    print(f"  routing_acc: T=0 = {dm:.3f} +/- {dh:.3f};  best T>0 = {bT} -> {bm:.3f} +/- {bh:.3f}")
    # dead-expert / hog check at T=0 vs default
    me0, _ = mean_ci([r["min_share"] for r in by_T2[0.0]])
    mx0, _ = mean_ci([r["max_share"] for r in by_T2[0.0]])
    print(f"  load balance at T=0: min module win-share={me0:.3f} (0=dead expert), "
          f"max share={mx0:.3f} (1=hog)")
    print(f"  VERDICT axis-2 (routing_acc): {lab}")

    # ---- Axis 3 -------------------------------------------------------------------------------
    print("\n[AXIS 3] Stage-3 first-task forgetA  (lower is better; train at T_floor, eval at T=0)")
    fz, pl = run_axis3()
    for label, by_T in (("GRAIL-fuse", fz), ("always-plastic", pl)):
        print(f"  -- {label} --")
        print("        metric  " + "".join(f"{('T='+str(T)):>19}" for T in T_GRID))
        for metric in ("forgetA", "errA_afterA"):
            cells = "".join(f"{fmt_ci([r[metric] for r in by_T[T]], 4):>19}" for T in T_GRID)
            print(f"  {metric:>12}  {cells}")
    dm, dh, bT, bm, bh, lab = _verdict(fz, "forgetA", lower_is_better=True)
    print(f"  forgetA (fuse): T=0 = {dm:.4f} +/- {dh:.4f};  best T>0 = {bT} -> {bm:.4f} +/- {bh:.4f}")
    # monotonic-hurt check across the grid (does forgetA grow with T?)
    means = [mean_ci([r["forgetA"] for r in fz[T]])[0] for T in T_GRID]
    mono_up = all(means[i] <= means[i + 1] + 1e-9 for i in range(len(means) - 1))
    print(f"  forgetA means by T {tuple(round(m,4) for m in means)}; "
          f"monotonically WORSE as T rises: {mono_up}")
    print(f"  VERDICT axis-3 (fuse forgetA): {lab}")

    print("\n" + "=" * 92)
    print("Read the three VERDICT lines above. HELPS = noise CI-separates better than T=0;")
    print("NEUTRAL = deterministic settling is just as good (challenges Pillar-4 load-bearing claim).")
    print("=" * 92)


if __name__ == "__main__":
    main()
