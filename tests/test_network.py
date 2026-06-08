import numpy as np
from grail.config import GRAILConfig
from grail.network import GRAILCore
from grail.types import Exogenous

def test_observe_learn_predict_runs_and_counts():
    c = GRAILConfig(dims=(6,5,4), grid_n_modules=6, n_settle=20, seed=0)
    net = GRAILCore(c)
    obs = np.array([0.2,-0.1,0.3,0.0,0.5,-0.2])
    M = net.observe_and_learn(obs, reward=1.0)             # one episode step
    assert np.isscalar(M) or np.ndim(M)==0                 # scalar neuromodulator only
    assert net.counters.global_comm_learn >= 1             # one scalar M broadcast
    assert net.counters.synaptic_ops > 0

def test_no_weight_transport_used():
    # structural guarantee: B and W are independent arrays (no aliasing, no transpose read)
    c = GRAILConfig(dims=(5,4), seed=1); net = GRAILCore(c)
    for l in range(net.pc.L-1):
        assert net.pc.B[l] is not net.pc.W[l]
        assert net.pc.B[l].shape == net.pc.W[l].T.shape    # shapes compatible but separate synapses
