import pytest
import numpy as np
import torch
import threading
import time

from cerebrum.config import CerebrumConfig
from cerebrum.unified import CerebrumNet
from cerebrum.grounding import MotorProcessor, System1Reflex
from cerebrum.types import Exogenous

@pytest.fixture
def test_config():
    return CerebrumConfig(
        dims=(4, 8),
        n_settle=2,
        seed=42,
        align_feedback=True,  # Enable Kolen-Pollack alignment to test the gap in CerebrumNet
        lam_kp=1e-2
    )

def test_cerebrum_net_align_feedback_gap(test_config):
    """
    Verify that Kolen-Pollack updates are applied to weights and feedback weights
    when align_feedback is True, and standard Hebbian updates are applied when False.
    """
    from dataclasses import replace
    # 1. Run with align_feedback=True
    cfg_true = replace(test_config, align_feedback=True, lam_kp=1e-2)
    net_true = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg_true)
    
    # 2. Run with align_feedback=False
    cfg_false = replace(test_config, align_feedback=False, lam_kp=1e-2)
    net_false = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg_false)
    
    # Force exact weight parity at init
    for m1, m2 in zip(net_true.modules, net_false.modules):
        for l in range(m1.L - 1):
            m2.W[l].data.copy_(m1.W[l].data)
            m2.B[l].data.copy_(m1.B[l].data)
            m2.Pi[l].copy_(m1.Pi[l])
            
    obs_slices = [np.ones(4), np.ones(4)]
    action = Exogenous(np.ones(2))
    
    # Step both
    net_true.step(obs_slices, action, reward=1.0)
    net_false.step(obs_slices, action, reward=1.0)
    
    # Verify that the resulting weights are different because True uses KP while False uses Hebbian
    for m1, m2 in zip(net_true.modules, net_false.modules):
        for l in range(m1.L - 1):
            assert not torch.allclose(m1.W[l], m2.W[l]), "W should differ when align_feedback is True vs False"
            assert not torch.allclose(m1.B[l], m2.B[l]), "B should differ when align_feedback is True vs False"

def test_system1_reflex_index_error_crash():
    """
    Verify that System1Reflex.evaluate raises ValueError instead of IndexError
    when passed a list or array with length < 3.
    """
    reflex = System1Reflex()
    with pytest.raises(ValueError, match="Sensory state sequence must have at least 3 elements."):
        reflex.evaluate([1.0])

def test_motor_processor_type_error_crash():
    """
    Verify that MotorProcessor.process catches TypeError/AttributeError and
    returns [0.0, 0.0] on failure instead of crashing.
    """
    mp = MotorProcessor(mode="linear", W_motor=[{"a": 1.0}])
    vels = mp.process([1.0])
    assert np.allclose(vels, [0.0, 0.0])

def test_cerebrum_net_obs_slices_string_crash(test_config):
    """
    Verify that CerebrumNet.step crashes with TypeError when obs_slices
    contains string elements (e.g. ['invalid', 'data']).
    """
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=test_config)
    obs_slices = [["invalid", "data"], np.ones(4)]
    action = Exogenous(np.ones(2))
    with pytest.raises(TypeError):
        net.step(obs_slices, action, reward=1.0)

def test_cerebrum_net_obs_slices_shape_mismatch_crash(test_config):
    """
    Verify that CerebrumNet.step raises a shape mismatch ValueError
    when obs_slices elements do not match slice_dim.
    """
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=test_config)
    obs_slices = [np.ones(3), np.ones(4)]  # First module gets size 3 instead of 4
    action = Exogenous(np.ones(2))
    with pytest.raises(ValueError):
        net.step(obs_slices, action, reward=1.0)

def test_settle_only_thread_unsafe_race(test_config):
    """
    Verify that CerebrumNet.settle_only acquires the thread lock (self._lock),
    protecting it against concurrent execution races.
    """
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=test_config)
    
    import inspect
    func = getattr(net, "settle_only_original", net.settle_only)
    source = inspect.getsource(func)
    assert "with self._lock" in source, "settle_only should contain lock acquisition"
