import numpy as np
from cerebrum.config import CerebrumConfig
from cerebrum.plasticity import Eligibility, weight_update, precision_update

def test_eligibility_lowpasses_presynaptic_activity():
    e = Eligibility(shape=(3,), cfg=CerebrumConfig(tau_e=4.0))
    for _ in range(1000): e.step(a_pre=np.ones(3))
    assert np.allclose(e.value.cpu().numpy(), 1.0, atol=1e-2)   # converges to steady presyn activity

def test_weight_update_matches_negative_grad_in_deterministic_limit():
    # -dF/dW_{ij} = Pi_i * eps_i * a_j  (precision-once). With M=theta=1 the rule must equal eta * that.
    c = CerebrumConfig(eta_w=1.0)
    Pi = np.array([2.0, 0.5]); eps = np.array([0.3, -0.4]); e = np.array([1.0, 0.5, -1.0])
    dW = weight_update(M=1.0, theta=np.ones((2,3)), Pi_post=Pi, eps_post=eps, elig=e, eta=c.eta_w)
    expected = np.outer(Pi*eps, e)                # (2,3)
    assert np.allclose(np.asarray(dW), expected)

def test_M_zero_means_no_learning():
    dW = weight_update(M=0.0, theta=np.ones((2,3)), Pi_post=np.ones(2),
                       eps_post=np.ones(2), elig=np.ones(3), eta=1.0)
    assert np.allclose(np.asarray(dW), 0.0)                    # neuromodulator gates WHEN to learn

def test_precision_converges_to_inverse_variance():
    c = CerebrumConfig(tau_pi=1.0, sigma0=0.0, kappa_pi=1.0)
    Pi = np.array([1.0]); var = 0.25
    for _ in range(5000): Pi = np.asarray(precision_update(Pi, eps_sq=np.array([var]), cfg=c))
    assert abs(Pi[0] - 1.0/var) < 0.1             # Pi -> 1/<eps^2>

def test_feedback_update_is_local_outer_product():
    from cerebrum.plasticity import feedback_update
    from cerebrum.config import CerebrumConfig
    c = CerebrumConfig(eta_b=1.0, lam_b=0.0)
    a_up = np.array([1.0, -1.0]); eps = np.array([0.5, 0.2, -0.3])   # B shape (2,3)
    B = np.zeros((2,3))
    dB = feedback_update(B, a_up=a_up, eps=eps, cfg=c)
    assert np.allclose(np.asarray(dB), np.outer(a_up, eps))    # uses only local a_up, eps (no W, no transpose)

def test_kp_feedback_update_is_transpose_of_forward_product():
    """KP rule: B's increment must be the EXACT transpose of W's four-factor increment so that
    (W - B.T) shrinks. Verified with matched eta, M and zero decay -> dB == (dW).T."""
    from cerebrum.plasticity import weight_update, feedback_update_kp
    Pi = np.array([2.0, 0.5, 1.0]); eps = np.array([0.3, -0.4, 0.1])   # post, len = d[l] = 3
    elig = np.array([1.0, 0.5])                                        # pre,  len = d[l+1] = 2
    dW = weight_update(M=1.0, theta=np.ones((3, 2)), Pi_post=Pi, eps_post=eps, elig=elig, eta=0.7)
    dB = feedback_update_kp(np.zeros((2, 3)), M=1.0, Pi_post=Pi, eps_post=eps,
                            elig=elig, eta=0.7, lam_kp=0.0)
    assert np.allclose(np.asarray(dB), np.asarray(dW).T)          # B gets the exact transpose of W's local product

def test_kp_feedback_update_uses_no_weight_transport():
    """KP rule reads only LOCAL signals (M, Pi_post*eps_post, elig); never W or W.T."""
    import inspect
    from cerebrum.plasticity import feedback_update_kp
    sig = set(inspect.signature(feedback_update_kp).parameters)
    assert "W" not in sig and "W_T" not in sig and "Wt" not in sig   # no weight argument at all

def test_kp_drives_B_toward_W_transpose():
    """Under repeated identical local pre/post and a MATCHED decay applied to both, B.T -> W
    (Kolen-Pollack). Start B,W mismatched; iterate the coupled update; cosine(B.T, W) -> ~1."""
    from cerebrum.plasticity import weight_update, feedback_update_kp
    rng = np.random.default_rng(0)
    W = 0.5 * rng.standard_normal((4, 3))         # (d[l], d[l+1])
    B = 0.5 * rng.standard_normal((3, 4))         # (d[l+1], d[l]) -- independent, mismatched
    Pi = np.array([1.0, 1.0, 1.0, 1.0]); eps = np.array([0.4, -0.3, 0.2, 0.1])
    elig = np.array([0.6, -0.2, 0.5]); eta = 0.05; lam = 0.02

    def cos(a, b):
        a = a.ravel(); b = b.ravel()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    c0 = cos(B.T, W)
    for _ in range(2000):
        dW = np.asarray(weight_update(M=1.0, theta=np.ones_like(W), Pi_post=Pi, eps_post=eps, elig=elig, eta=eta))
        W = W + dW - lam * W
        dB = np.asarray(feedback_update_kp(B, M=1.0, Pi_post=Pi, eps_post=eps, elig=elig, eta=eta, lam_kp=lam))
        B = B + dB
    c1 = cos(B.T, W)
    assert c1 > 0.99 and c1 > c0      # B.T converges to W (alignment learned, not transported)
