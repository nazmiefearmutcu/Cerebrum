import numpy as np
from grail.config import GRAILConfig
from grail.pc_core import PCAreas
from grail.energy import (spike_sparsity, dynamic_synaptic_ops, dynamic_energy_magnitude,
                          dense_backprop_ops, global_comm_per_update)


def test_spike_sparsity_drops_as_error_shrinks():
    eps_hi = [np.array([1.0, -1.0, 0.5]), np.array([2.0, 0.3])]      # mostly active
    eps_lo = [np.array([1e-9, 0.0, 0.0]), np.array([0.0, 0.0])]      # mostly silent (competent)
    assert spike_sparsity(eps_hi, tol=1e-6) > spike_sparsity(eps_lo, tol=1e-6)
    assert spike_sparsity(eps_lo, tol=1e-6) < 0.2

def test_dynamic_ops_below_dense_when_sparse():
    cfg = GRAILConfig(dims=(6, 5, 4)); net = PCAreas(cfg)
    net.eps = [np.zeros(6), np.zeros(5), np.zeros(4)]
    net.eps[0][0] = 1.0                                             # a single active error unit
    assert dynamic_synaptic_ops(net, tol=1e-6) < dense_backprop_ops(cfg.dims)

def test_dynamic_ops_grow_with_activity():
    cfg = GRAILConfig(dims=(6, 5, 4)); net = PCAreas(cfg)
    net.eps = [np.ones(6), np.zeros(5), np.zeros(4)]                # 6 active in area 0
    many = dynamic_synaptic_ops(net, tol=1e-6)
    net.eps = [np.zeros(6), np.zeros(5), np.zeros(4)]; net.eps[0][0] = 1.0
    few = dynamic_synaptic_ops(net, tol=1e-6)
    assert many > few                                              # more spikes -> more dynamic ops

def test_dynamic_energy_magnitude_decays_with_error():
    cfg = GRAILConfig(dims=(6, 5, 4)); net = PCAreas(cfg)
    net.eps = [np.ones(6), np.zeros(5), np.zeros(4)]; hi = dynamic_energy_magnitude(net)
    net.eps = [0.01 * np.ones(6), np.zeros(5), np.zeros(4)]; lo = dynamic_energy_magnitude(net)
    assert 0.0 < lo < hi                                           # smaller error -> less switching energy

def test_global_comm_learn_is_scalar_vs_backprop_vectors():
    g = global_comm_per_update(dims=(10, 8, 6))
    assert g["grail_learn_scalars"] == 1                           # one scalar M
    assert g["backprop_error_vector_elems"] == 8 + 6               # O(depth) error-vector elements
    assert g["grail_learn_scalars"] < g["backprop_error_vector_elems"]

def test_energy_curve_decays_with_competence():
    from benchmarks.run_energy import run_energy
    curve, cfg = run_energy(seed=0, passes=150, measure_every=50)
    # curve rows = (pass, recon_err, eps_sparsity, dyn_ops, dyn_energy)
    assert curve[-1][1] < curve[0][1] * 0.8     # competence rises: reconstruction error drops >=20%
    assert curve[-1][4] < curve[0][4]           # dynamic switching energy decays with competence
