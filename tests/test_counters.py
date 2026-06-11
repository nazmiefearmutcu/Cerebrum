import numpy as np
from cerebrum.counters import Counters

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

def test_dense_vs_dynamic_ops_recording():
    c = Counters()
    c.record_synaptic_ops(dense=100, dynamic=20)
    assert c.dense_synaptic_ops == 100
    assert c.dynamic_synaptic_ops == 20
    assert c.synaptic_ops == 20  # synaptic_ops alias tracks dynamic ops

def test_pc_core_sparsity_excludes_x0():
    from cerebrum.config import CerebrumConfig
    from cerebrum.pc_core import PCAreas
    from cerebrum.rng import SeededRNG
    
    cfg = CerebrumConfig(dims=(5, 8, 8), seed=0)
    net = PCAreas(cfg)
    c = Counters()
    
    # Set x[0] to all non-zero, x[1] and x[2] to all zero
    net.x[0] = np.ones(5)
    net.x[1] = np.zeros(8)
    net.x[2] = np.zeros(8)
    
    rng = SeededRNG(0)
    net.settle_step(rng, T=0.0, clamp_bottom=np.ones(5), counters=c)
    
    # x[0] (5 units) is excluded. Total recorded units should be exactly 8 + 8 = 16.
    assert c._total == 16
