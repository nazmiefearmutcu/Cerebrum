import numpy as np
from grail.counters import Counters

def test_counts_learn_and_infer_separately():
    c = Counters()
    c.record_global_learn(1)          # one scalar M broadcast at learn time
    c.record_global_infer_vectors(k=4, width=8)  # broadcast of 4 slots x width 8 at infer time
    c.record_synaptic_ops(100)
    assert c.global_comm_learn == 1
    assert c.global_comm_infer == 4*8
    assert c.synaptic_ops == 100

def test_sparsity_tracks_active_fraction():
    c = Counters()
    c.record_activity(np.array([0.0, 0.0, 1.0, 0.0]))  # 1/4 active
    assert abs(c.sparsity() - 0.25) < 1e-9
