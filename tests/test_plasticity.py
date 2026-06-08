import numpy as np
from grail.config import GRAILConfig
from grail.plasticity import Eligibility, weight_update, precision_update

def test_eligibility_lowpasses_presynaptic_activity():
    e = Eligibility(shape=(3,), cfg=GRAILConfig(tau_e=4.0))
    for _ in range(1000): e.step(a_pre=np.ones(3))
    assert np.allclose(e.value, 1.0, atol=1e-2)   # converges to steady presyn activity

def test_weight_update_matches_negative_grad_in_deterministic_limit():
    # -dF/dW_{ij} = Pi_i * eps_i * a_j  (precision-once). With M=theta=1 the rule must equal eta * that.
    c = GRAILConfig(eta_w=1.0)
    Pi = np.array([2.0, 0.5]); eps = np.array([0.3, -0.4]); e = np.array([1.0, 0.5, -1.0])
    dW = weight_update(M=1.0, theta=np.ones((2,3)), Pi_post=Pi, eps_post=eps, elig=e, eta=c.eta_w)
    expected = np.outer(Pi*eps, e)                # (2,3)
    assert np.allclose(dW, expected)

def test_M_zero_means_no_learning():
    dW = weight_update(M=0.0, theta=np.ones((2,3)), Pi_post=np.ones(2),
                       eps_post=np.ones(2), elig=np.ones(3), eta=1.0)
    assert np.allclose(dW, 0.0)                    # neuromodulator gates WHEN to learn

def test_precision_converges_to_inverse_variance():
    c = GRAILConfig(tau_pi=1.0, sigma0=0.0, kappa_pi=1.0)
    Pi = np.array([1.0]); var = 0.25
    for _ in range(5000): Pi = precision_update(Pi, eps_sq=np.array([var]), cfg=c)
    assert abs(Pi[0] - 1.0/var) < 0.1             # Pi -> 1/<eps^2>

def test_feedback_update_is_local_outer_product():
    from grail.plasticity import feedback_update
    from grail.config import GRAILConfig
    c = GRAILConfig(eta_b=1.0, lam_b=0.0)
    a_up = np.array([1.0, -1.0]); eps = np.array([0.5, 0.2, -0.3])   # B shape (2,3)
    B = np.zeros((2,3))
    dB = feedback_update(B, a_up=a_up, eps=eps, cfg=c)
    assert np.allclose(dB, np.outer(a_up, eps))    # uses only local a_up, eps (no W, no transpose)
