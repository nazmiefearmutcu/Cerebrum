"""Fast smoke test for the Pillar-4 settling-noise ablation (benchmarks/run_pillar4_ablation.py).

Asserts only ROBUSTLY-TRUE things:
  - every axis returns finite, in-range numbers for both T=0 (deterministic) and T>0;
  - T_floor is actually threaded into the settling noise scale (a mechanism check, not a
    performance claim) — at T=0 the SeededRNG-driven settle is deterministic, at T>0 it is not;
  - the Task-1 readout is structurally INVARIANT to T_floor (the completion path bypasses PC
    settling) — a bit-exact null we are confident about.

It does NOT assert that noise helps any axis; whether noise helps is the empirical question the
full sweep answers, and a null there is a legitimate result.
"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.run_pillar4_ablation import task1_acc, stage2_routing, stage3_continual
from cerebrum.pc_core import PCAreas
from cerebrum.config import CerebrumConfig
from cerebrum.rng import SeededRNG


def test_settle_noise_scale_tracks_T_floor():
    """The Langevin term sqrt(2 T dt/tau_x) dW must be ZERO at T=0 and NON-zero at T>0,
    holding the rng stream fixed. This is the mechanism the ablation toggles."""
    cfg = CerebrumConfig(dims=(4, 4))
    obs = np.ones(4)

    def settled_top(T):
        net = PCAreas(cfg)
        rng = SeededRNG(0)                       # same seed/stream both calls
        for _ in range(5):
            net.settle_step(rng, T=T, clamp_bottom=obs)
        return net.x[-1].copy()

    x_det = settled_top(0.0)
    x_det2 = settled_top(0.0)
    x_noisy = settled_top(0.05)
    assert np.allclose(x_det, x_det2)            # T=0 is deterministic (reproducible)
    assert not np.allclose(x_det, x_noisy)       # T>0 actually injects noise into the settle
    assert np.all(np.isfinite(np.asarray(x_noisy)))


def test_axis1_finite_and_structurally_invariant():
    """Task-1 accuracy is finite, in [0,1], and BIT-EXACT across T_floor (readout bypasses PC
    settling). This null is robust — we assert it."""
    accs = {T: task1_acc(T, K=5, seed=0) for T in (0.0, 0.02, 0.2)}
    for a in accs.values():
        assert np.isfinite(a) and 0.0 <= a <= 1.0
    assert accs[0.0] == accs[0.02] == accs[0.2]   # exact null


def test_axis2_routing_finite_and_balanced_at_T0():
    """Stage-2 routing returns finite metrics in range for both deterministic and noisy settle.
    Robust load-balance fact: at T=0 there is no DEAD expert (every module wins some slots)."""
    for T in (0.0, 0.02):
        r = stage2_routing(T, n_modules=4, trials=120, seed=0)
        assert np.isfinite(r["routing_acc"]) and 0.0 <= r["routing_acc"] <= 1.0
        assert np.isfinite(r["win_entropy"]) and r["win_entropy"] >= 0.0
        assert 0.0 <= r["min_share"] <= r["max_share"] <= 1.0
    # determinism: no settling noise => no dead expert here (robust on this balanced task)
    r0 = stage2_routing(0.0, n_modules=4, trials=120, seed=0)
    assert r0["min_share"] > 0.0, "T=0 produced a dead expert on the balanced routing task"


def test_axis3_forgetA_finite_both_arms():
    """Stage-3 forgetA is finite for fuse and always-plastic, at deterministic and noisy settle.
    Eval is T=0 by design, so this is a pure function of the learned weights."""
    for use_fuse in (True, False):
        for T in (0.0, 0.05):
            r = stage3_continual(T, use_fuse=use_fuse, seed=0, passes=20)  # short for speed
            assert np.isfinite(r["forgetA"])
            assert np.isfinite(r["errA_afterA"]) and r["errA_afterA"] >= 0.0
