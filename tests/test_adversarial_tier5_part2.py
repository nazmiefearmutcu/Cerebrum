import pytest
import numpy as np
import torch

from cerebrum.config import CerebrumConfig
from cerebrum.unified import CerebrumNet
from cerebrum.grounding import SensoryProcessor, MotorProcessor, System1Reflex, CerebrumROSNode, MockPyBullet, std_msgs
from cerebrum.types import Exogenous
from cerebrum.metaplasticity import MetaplasticFuse

@pytest.fixture
def base_config():
    return CerebrumConfig(dims=(4, 8), n_settle=2, seed=42)

def test_mock_pybullet_load_urdf_invalid_orientation_adversarial():
    """Verify that MockPyBullet fails cleanly or crashes as expected when loadURDF orientation lacks 4 elements."""
    p = MockPyBullet()
    p.connect(p.DIRECT)
    
    with pytest.raises(ValueError, match="Orientation must be a 4-element quaternion."):
        p.loadURDF("robot.urdf", baseOrientation=(0.0, 0.0, 0.0))
        
    p.disconnect()

def test_system1_reflex_evaluate_insufficient_length_adversarial():
    """Verify that System1Reflex evaluate() raises a ValueError on list inputs with fewer than 3 elements."""
    reflex = System1Reflex()
    
    with pytest.raises(ValueError, match="Sensory state sequence must have at least 3 elements."):
        reflex.evaluate([0.15, 0.0])

def test_motor_processor_nan_in_weights_adversarial():
    """Verify that MotorProcessor sanitizes NaNs to [0.0, 0.0]."""
    W_nan = np.array([[np.nan, 1.0], [1.0, 1.0]])
    mp = MotorProcessor(mode="linear", W_motor=W_nan)
    
    vels = mp.process([1.0, 1.0])
    assert not np.isnan(vels).any(), "NaNs in W_motor should be sanitized."
    assert np.allclose(vels, [0.0, 0.0])

def test_metaplasticity_numerical_stability_adversarial(base_config):
    """Verify MetaplasticFuse remains numerically stable with extremely large negative surprises (underflows to 0.0)."""
    fuse = MetaplasticFuse((2, 2), base_config)
    
    # Mock high baseline surprise with zero raw surprise to produce extreme negative surprise difference
    fuse.S_bar = torch.full((2, 2), 1e8, dtype=torch.float64)
    
    # Update with zero inputs
    theta = fuse.update(torch.zeros(2), torch.zeros(2), torch.zeros(2))
    
    assert torch.isfinite(theta).all(), "Metaplasticity weights should not become NaN under extreme negative surprise."
    assert torch.allclose(theta, torch.zeros_like(theta)), "Extreme negative surprise should freeze weights (theta -> 0.0)."

def test_ros_node_sensory_len_5_misinterpretation_adversarial(base_config):
    """Verify the len=5 heuristic vulnerability is prevented by default (is_direct_state=False)."""
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    reflex = System1Reflex(collision_threshold=0.20, tilt_threshold=0.5)
    node = CerebrumROSNode(net, node_name="len_5_node", reflex=reflex, is_direct_state=False)
    
    # Message data of length 5 where index 1 represents camera but is interpreted as tilt > threshold (0.5)
    msg = std_msgs.msg.Float64MultiArray()
    msg.data = [1.0, 0.8, 0.0, 0.0, 0.0]
    
    node.sensory_callback(msg)
    
    pub = node.publishers["/motor_commands"]
    if len(pub.published_messages) > 0:
        # It should not trigger the tilt reflex STABILIZE maneuver [-1.0, -1.0]
        assert not np.allclose(pub.published_messages[-1].data, [-1.0, -1.0])
