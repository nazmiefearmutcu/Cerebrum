import numpy as np, pytest
from cerebrum.config import CerebrumConfig
from cerebrum.grid_head import GridHead
from cerebrum.types import Exogenous

def test_code_shape_and_unit_norm_per_module():
    gh = GridHead(CerebrumConfig(grid_n_modules=6)); gh.reset()
    g = gh.encode(); assert g.shape == (12,)              # 6 modules x 2
    mods = g.reshape(6,2)
    assert np.allclose(np.linalg.norm(mods, axis=1), 1.0) # each module is a unit phasor

def test_path_integration_loop_closure():
    gh = GridHead(CerebrumConfig()); gh.reset()
    g0 = gh.encode().copy()
    for a in [[1,0],[0,1],[-1,0],[0,-1]]:                 # walk a unit square, return to start
        gh.transition(Exogenous(np.array(a, float)))
    assert np.allclose(gh.encode(), g0, atol=1e-9)        # exact loop closure (structural)

def test_path_integration_is_additive():
    gh = GridHead(CerebrumConfig()); gh.reset()
    gh.transition(Exogenous(np.array([2.0,1.0]))); gA = gh.encode().copy()
    gh.reset()
    gh.transition(Exogenous(np.array([1.0,0.0]))); gh.transition(Exogenous(np.array([1.0,1.0])))
    assert np.allclose(gh.encode(), gA, atol=1e-9)        # displacement composes (graph algebra)

def test_transition_rejects_plain_array():
    gh = GridHead(CerebrumConfig()); gh.reset()
    with pytest.raises(TypeError):
        gh.transition(np.array([1.0,0.0]))                # data-derived action = BAN-1

def test_bind_then_complete_recovers_observation_at_bound_location():
    gh = GridHead(CerebrumConfig()); gh.reset()
    obs = np.array([0.0, 1.0, 0.0, -1.0])
    gh.bind(obs)                                  # bind obs at current location
    rec = gh.complete()                           # complete at the SAME location
    # cosine similarity high (single binding -> proportional recall)
    assert np.dot(rec, obs)/(np.linalg.norm(rec)*np.linalg.norm(obs)+1e-9) > 0.9

def test_completion_generalizes_to_path_integrated_location():
    gh = GridHead(CerebrumConfig()); gh.reset()
    obsA = np.array([1.0,0.0,0.0]); obsB = np.array([0.0,1.0,0.0])
    gh.bind(obsA)
    gh.transition(__import__('cerebrum.types',fromlist=['Exogenous']).Exogenous(np.array([3.0,2.0])))
    gh.bind(obsB)
    # return to A by exact inverse displacement; completion should recall obsA (graph completion)
    gh.transition(__import__('cerebrum.types',fromlist=['Exogenous']).Exogenous(np.array([-3.0,-2.0])))
    rec = gh.complete()
    assert np.dot(rec, obsA)/(np.linalg.norm(rec)*np.linalg.norm(obsA)+1e-9) > 0.8
