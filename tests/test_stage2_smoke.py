import numpy as np
from benchmarks.tasks.binding import run_binding

def test_routing_accuracy_rises_above_chance():
    res = run_binding(n_modules=4, k_slots=1, trials=400, seed=0)
    assert res["routing_acc"] > 1.0/4 + 0.1            # gate learns to route the target above chance
    assert res["win_entropy"] > 0.1                     # load is balanced (no single dead/hog collapse)

def test_soft_mixer_collapses_to_undifferentiated_mixing():
    from benchmarks.baselines.soft_mixer import run_binding_soft
    from benchmarks.tasks.binding import run_binding
    hard = run_binding(n_modules=4, k_slots=1, trials=400, seed=0)
    soft = run_binding_soft(n_modules=4, k_slots=1, trials=400, seed=0)
    # soft aggregation mixes all modules' content every step -> cannot route selectively
    assert soft["routing_acc"] < hard["routing_acc"]
    assert soft["mean_slot_participation"] > 1.5        # >1 module contributes per slot (continuous mixing, not one-hot)
