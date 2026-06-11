import numpy as np
from cerebrum.config import CerebrumConfig
from cerebrum.network2 import CerebrumWorkspaceNet
from cerebrum.invariants import assert_one_hot

def test_step_runs_routes_and_counts():
    cfg = CerebrumConfig(dims=(4,4), n_settle=8, seed=0)
    net = CerebrumWorkspaceNet(n_modules=3, k_slots=2, slice_dim=4, cfg=cfg)
    obs = [np.array([1.,0,0,0]), np.array([0,1.,0,0]), np.array([0,0,1.,0])]
    z, M = net.step(obs, reward=1.0)
    assert_one_hot(z, axis=0)                            # routing decision is one-hot
    assert np.ndim(M) == 0                               # scalar neuromodulator
    assert net.counters.global_comm_infer > 0           # broadcast vectors counted at infer time

def test_broadcast_influences_modules():
    cfg = CerebrumConfig(dims=(4,4), n_settle=8, seed=1)
    net = CerebrumWorkspaceNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg)
    obs = [np.array([1.,0,0,0]), np.array([0,0,0,1.])]
    net.step(obs, reward=1.0)
    assert np.any(net.workspace.slots != 0)              # a winner wrote content that will broadcast next step
