import numpy as np
from grail.config import GRAILConfig
from grail.pc_core import PCAreas
from grail.nonlinear import g_act

def make():
    c = GRAILConfig(dims=(4,3,2), seed=0)
    return PCAreas(c), c

def test_shapes():
    pc, c = make()
    assert len(pc.x) == 3 and pc.x[0].shape == (4,)
    assert len(pc.W) == 2 and pc.W[0].shape == (4,3)   # W[l]: predicts area l (size dims[l]) from area l+1 (size dims[l+1])
    assert pc.Pi[0].shape == (4,)                       # diagonal precision = vector

def test_error_is_input_minus_prediction():
    pc, c = make()
    pc.x[1][:] = np.array([0.5, -0.5, 0.2])
    pc.x[0][:] = np.array([0.1, 0.1, 0.1, 0.1])
    pc.compute_errors(top_pred=np.zeros(2))
    yhat0 = g_act(pc.W[0] @ pc.x[1])
    assert np.allclose(pc.eps[0], pc.x[0] - yhat0)

def test_energy_decreases_when_error_decreases():
    pc, c = make()
    pc.compute_errors(top_pred=np.zeros(2)); e_hi = pc.energy()
    for l in range(len(pc.x)): pc.x[l][:] = 0.0
    for l in range(len(pc.W)): pc.W[l][:] = 0.0
    pc.compute_errors(top_pred=np.zeros(2)); e_lo = pc.energy()
    assert e_lo <= e_hi  # zero error -> lower precision-weighted energy term
