"""Energy / operation accounting for GRAIL — success-axis-2 instrumentation.

The honest energy story (spec section neuromorphic_mapping): only DYNAMIC switching energy decays
with competence (as prediction error eps -> 0, error neurons fall silent and stop driving their
synapses). Learn-time global communication is O(1) — a single scalar neuromodulator M — whereas a
matched backprop network broadcasts an error VECTOR per layer (O(depth) elements). Static/leakage
power and settle-time energy do NOT decay; only the event-driven dynamic term does."""
import numpy as np


def spike_sparsity(eps_list, tol=1e-6):
    """Fraction of error-neuron units that are ACTIVE (|eps| > tol). Event-driven: well-predicted
    units are silent, so this falls toward a floor as the network becomes competent."""
    active = sum(int(np.sum(np.abs(e) > tol)) for e in eps_list)
    total = sum(int(e.size) for e in eps_list)
    return active / total if total else 0.0


def dynamic_synaptic_ops(net, tol=1e-6):
    """Event-driven synaptic-op count: a forward synapse computes only when its postsynaptic error
    neuron spikes. ops = sum_l (#active eps_l) * fan-in. Silent error neurons cost ~0, so the
    dynamic op count decays with competence."""
    ops = 0
    for l in range(net.L - 1):
        active = int(np.sum(np.abs(net.eps[l]) > tol))
        ops += active * net.W[l].shape[1]      # each active error unit drives its fan-in synapses
    return ops


def dynamic_energy_magnitude(net):
    """Magnitude-weighted dynamic switching-energy proxy: sum over predicted areas of (total error
    activity Σ|eps_l|) * fan-in. In graded event-driven coding the switching energy scales with total
    error activity, so this decays SMOOTHLY as the network becomes competent (eps -> 0). It is the
    robust headline energy metric; the thresholded spike count is a conservative companion."""
    e = 0.0
    for l in range(net.L - 1):
        e += float(np.sum(np.abs(net.eps[l]))) * net.W[l].shape[1]
    return e


def dense_backprop_ops(dims):
    """Dense forward+backward MAC count for a matched backprop net: every synapse computes every
    step (rho = 1), forward AND backward. The comparator GRAIL's event-driven sparsity undercuts."""
    fwd = sum(dims[l] * dims[l + 1] for l in range(len(dims) - 1))
    return 2 * fwd     # forward + backward dense passes


def global_comm_per_update(dims):
    """Global-communication events crossing the whole network per WEIGHT UPDATE.
    GRAIL: ONE scalar neuromodulator M (a single diffuse wire). Backprop: an error VECTOR at every
    layer (O(depth) vector elements that must be transported between layers)."""
    return {
        "grail_learn_scalars": 1,
        "backprop_error_vector_elems": int(sum(dims[1:])),   # error vectors at each non-input layer
    }
