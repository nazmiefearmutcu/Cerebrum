import numpy as np
from benchmarks.tasks.binding import run_binding

def test_routing_accuracy_rises_above_chance():
    res = run_binding(n_modules=4, k_slots=1, trials=400, seed=0)
    assert res["routing_acc"] > 1.0/4 + 0.1            # gate learns to route the target above chance
    assert res["win_entropy"] > 0.1                     # load is balanced (no single dead/hog collapse)
