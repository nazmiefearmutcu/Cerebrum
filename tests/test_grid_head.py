import numpy as np, pytest
from grail.config import GRAILConfig
from grail.grid_head import GridHead
from grail.types import Exogenous

def test_code_shape_and_unit_norm_per_module():
    gh = GridHead(GRAILConfig(grid_n_modules=6)); gh.reset()
    g = gh.encode(); assert g.shape == (12,)              # 6 modules x 2
    mods = g.reshape(6,2)
    assert np.allclose(np.linalg.norm(mods, axis=1), 1.0) # each module is a unit phasor

def test_path_integration_loop_closure():
    gh = GridHead(GRAILConfig()); gh.reset()
    g0 = gh.encode().copy()
    for a in [[1,0],[0,1],[-1,0],[0,-1]]:                 # walk a unit square, return to start
        gh.transition(Exogenous(np.array(a, float)))
    assert np.allclose(gh.encode(), g0, atol=1e-9)        # exact loop closure (structural)

def test_path_integration_is_additive():
    gh = GridHead(GRAILConfig()); gh.reset()
    gh.transition(Exogenous(np.array([2.0,1.0]))); gA = gh.encode().copy()
    gh.reset()
    gh.transition(Exogenous(np.array([1.0,0.0]))); gh.transition(Exogenous(np.array([1.0,1.0])))
    assert np.allclose(gh.encode(), gA, atol=1e-9)        # displacement composes (graph algebra)

def test_transition_rejects_plain_array():
    gh = GridHead(GRAILConfig()); gh.reset()
    with pytest.raises(TypeError):
        gh.transition(np.array([1.0,0.0]))                # data-derived action = BAN-1
