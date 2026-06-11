import numpy as np
from cerebrum.nonlinear import g_act, g_deriv

def test_tanh_values():
    u = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(g_act(u), np.tanh(u))

def test_derivative_matches_finite_difference():
    u = np.linspace(-2, 2, 11); h = 1e-6
    fd = (g_act(u+h) - g_act(u-h)) / (2*h)
    assert np.allclose(g_deriv(u), fd, atol=1e-5)
