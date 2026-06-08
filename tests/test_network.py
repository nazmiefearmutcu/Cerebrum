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

def _mean_cos_BWt(net):
    cs = []
    for l in range(net.pc.L-1):
        a = net.pc.B[l].T.ravel(); b = net.pc.W[l].ravel()
        cs.append(float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b))))
    return float(np.mean(cs))

def test_align_feedback_flag_raises_BWt_alignment():
    # End-to-end: with align_feedback=True, B should track W.T far better than with it OFF.
    # KP needs lam_kp comparable to the effective weight rate (eta_w/tau_w), so use a regime
    # where the weight update is not glacial relative to the matched decay.
    obs = np.array([0.2,-0.1,0.3,0.0,0.5,-0.2])
    def train(flag):
        c = GRAILConfig(dims=(6,5,4), n_settle=20, seed=0, align_feedback=flag,
                        lam_kp=2e-3, tau_w=5.0, eta_w=0.1)
        net = GRAILCore(c)
        for _ in range(400):
            net.observe_and_learn(obs, reward=1.0)
        return _mean_cos_BWt(net)
    cos_off = train(False); cos_on = train(True)
    assert cos_on > cos_off + 0.2 and cos_on > 0.5   # alignment is LEARNED locally, not transported
