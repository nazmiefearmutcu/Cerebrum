import numpy as np
from cerebrum.config import CerebrumConfig
from cerebrum.pc_core import PCAreas
from cerebrum.nonlinear import g_act

def make():
    c = CerebrumConfig(dims=(4,3,2), seed=0)
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

def test_balance_grid_precision_off_is_identity():
    # flag OFF: predict(top) returns the raw top_pred unchanged (default behavior preserved)
    pc, c = make()
    pc.x[0][:] = np.array([0.1, 0.2, -0.1, 0.3])
    pc.x[1][:] = np.array([0.2, -0.1, 0.05])
    pc.compute_errors(top_pred=np.zeros(2))           # populate eps[L-2]
    big = np.array([40.0, -50.0])                     # grid-scale prediction
    assert np.allclose(pc.predict(pc.L - 1, top_pred=big), big)

def test_balance_grid_precision_on_downscales_dominating_pred():
    # flag ON: a huge top_pred is scaled DOWN to the top-area bottom-up signal scale; a small one isn't
    c = CerebrumConfig(dims=(4, 3, 2), seed=0, balance_grid_precision=True)
    pc = PCAreas(c)
    pc.x[0][:] = np.array([0.1, 0.2, -0.1, 0.3])
    pc.x[1][:] = np.array([0.2, -0.1, 0.05])
    pc.x[2][:] = np.array([0.05, -0.05])
    pc.compute_errors(top_pred=np.zeros(2))           # populate eps so bottom-up ref is defined
    ref = pc._bottomup_scale_top()
    big = np.array([40.0, -50.0])                     # grid-scale prediction (norm >> ref)
    out = pc.predict(pc.L - 1, top_pred=big)
    assert np.linalg.norm(out) < np.linalg.norm(big)              # was crushing -> down-weighted
    assert np.isclose(np.linalg.norm(out), ref, rtol=1e-6)        # matched to bottom-up scale
    small = np.array([1e-4, -1e-4])                   # already below ref -> never amplified
    assert np.allclose(pc.predict(pc.L - 1, top_pred=small), small)
