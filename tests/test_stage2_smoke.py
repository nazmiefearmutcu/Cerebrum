import numpy as np
from benchmarks.tasks.binding import run_binding

def test_routing_accuracy_rises_above_chance():
    res = run_binding(n_modules=4, k_slots=1, trials=400, seed=0)
    assert res["routing_acc"] > 1.0/4 + 0.1            # gate routes the target above chance
    assert res["win_entropy"] > 0.1                     # load is balanced (no single dead/hog collapse)

def test_m6_routing_well_above_chance_after_fix():
    res = run_binding(n_modules=6, k_slots=1, trials=500, seed=0)
    assert res["routing_acc"] > 0.5      # M=6 (chance 0.167): gate-decay + low temp recover bid routing

def test_soft_mixer_collapses_to_undifferentiated_mixing():
    from benchmarks.baselines.soft_mixer import run_binding_soft
    from benchmarks.tasks.binding import run_binding
    # The write-rule contrast is shown at a MODERATE gate temperature where the selection distribution P
    # spreads: a near-degenerate low-temp P makes the soft write approximate the one-hot write, hiding the
    # mixing. Match the temperature for an apples-to-apples comparison (only the write rule differs).
    hard = run_binding(n_modules=4, k_slots=1, trials=400, seed=0, gate_temp=0.5)
    soft = run_binding_soft(n_modules=4, k_slots=1, trials=400, seed=0, gate_temp=0.5)
    assert soft["routing_acc"] < hard["routing_acc"]    # soft blend cannot route as cleanly as one-hot
    assert soft["mean_slot_participation"] > 1.5        # >1 module contributes per slot (continuous mixing)
