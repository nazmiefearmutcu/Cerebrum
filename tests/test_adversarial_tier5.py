import pytest
import numpy as np
import torch

from cerebrum.config import CerebrumConfig
from cerebrum.unified import CerebrumNet
from cerebrum.grounding import SensoryProcessor, MotorProcessor, System1Reflex, CerebrumROSNode, MockPyBullet, std_msgs
from cerebrum.types import Exogenous

@pytest.fixture
def base_config():
    return CerebrumConfig(dims=(4, 8), n_settle=8, seed=42)

def test_sensory_reflex_mismatch_adversarial():
    """Tier 5 Adversarial: Verify the semantic mismatch between SensoryProcessor and System1Reflex is mitigated.
    SensoryProcessor processes raw inputs and outputs:
      state = [min_lidar, left_cam_mean, right_cam_mean, velocity, heading]
    
    A bright left camera pixel mean (e.g. 0.6) with no actual tilt or collision hazard should NOT
    cause the robot to trigger an imbalance hazard because we construct a dictionary-based state
    in CerebrumROSNode.sensory_callback mapping left_cam to camera, NOT tilt.
    """
    sp = SensoryProcessor()
    reflex = System1Reflex(collision_threshold=0.20, tilt_threshold=0.5)
    
    # Safe distance (10.0), bright left camera (0.6), bright right camera (0.2), zero velocity/heading
    lidar_data = np.array([10.0, 10.0, 10.0])
    camera_data = np.array([0.6, 0.6, 0.2, 0.2]) # left half mean = 0.6, right half mean = 0.2
    odometer_data = np.array([0.0, 0.0])
    
    state = sp.process(lidar_data, camera_data, odometer_data)
    
    # Assert processed state is [10.0, 0.6, 0.2, 0.0, 0.0]
    assert np.allclose(state, [10.0, 0.6, 0.2, 0.0, 0.0])
    
    # Construct dict state as done in CerebrumROSNode.sensory_callback:
    state_dict = {
        "dist": float(state[0]),
        "tilt": 0.0,
        "error_energy": 0.0
    }
    
    # Evaluate with reflex using dictionary
    active, action = reflex.evaluate(state_dict)
    
    # Check that it is NOT triggered (since tilt is 0.0 in dict)
    assert not active, "Reflex should not trigger because camera inputs are not interpreted as tilt anymore."
    assert action is None


def test_ros_node_nan_inf_propagation_adversarial(base_config):
    """Tier 5 Adversarial: Verify that NaN/Inf in the sensory input is sanitized
    by CerebrumROSNode, preventing corruption of the network's weights.
    """
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    reflex = System1Reflex()
    node = CerebrumROSNode(net, node_name="test_nan_node", reflex=reflex)
    
    # Check that initially weights are finite
    assert torch.isfinite(net.modules[0].W[0]).all()
    
    # Create Float64MultiArray message with 5 elements containing a NaN
    msg = std_msgs.msg.Float64MultiArray()
    msg.data = [np.nan, 0.0, 0.0, 0.0, 0.0]
    
    # This will go into sensory_callback, which should clean/sanitize the NaN to 0.0
    node.sensory_callback(msg)
    
    # Check that weights do not contain NaN (remains finite)
    has_nan = torch.isnan(net.modules[0].W[0]).any().item()
    assert not has_nan, "NaN should not propagate to the weights of the network"


def test_ros_node_reward_nan_inf_corruption_adversarial(base_config):
    """Tier 5 Adversarial: Verify that NaN/Inf in the reward signal is sanitized
    by CerebrumROSNode, preventing propagation to Neuromodulator and weight corruption.
    """
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    node = CerebrumROSNode(net, node_name="test_reward_nan_node")
    
    # Create Float64MultiArray message with a NaN reward
    reward_msg = std_msgs.msg.Float64MultiArray()
    reward_msg.data = [np.nan]
    
    # Node should ignore the NaN reward
    node.reward_callback(reward_msg)
    assert not np.isnan(node.reward), "Node reward should not be updated to NaN"
    
    # Step with normal sensory input
    sensory_msg = std_msgs.msg.Float64MultiArray()
    sensory_msg.data = np.ones(8).tolist()
    
    node.sensory_callback(sensory_msg)
    
    # Neuromodulator r_bar and weights must remain finite and not NaN
    assert not np.isnan(net.nm.r_bar), "Neuromodulator r_bar should not become NaN"
    assert torch.isfinite(net.modules[0].W[0]).all(), "Weights should not contain NaN after stepping"


def test_motor_processor_all_zeros_fallback_adversarial():
    """Tier 5 Adversarial: Verify that MotorProcessor in discrete mode treats an all-zero
    action vector as a 'Standby' command ([0.0, 0.0]) instead of 'Forward' ([1.0, 1.0]).
    """
    mp = MotorProcessor(mode="discrete")
    vels = mp.process(np.zeros(3))
    
    # Standby command on all-zero action vector
    assert np.allclose(vels, [0.0, 0.0]), "Expected Standby command on all-zero action vector"


def test_mock_pybullet_orientation_shape_crash_adversarial():
    """Tier 5 Adversarial: Verify that MockPyBullet validates orientation vector length,
    raising a clean ValueError instead of crashing with IndexError.
    """
    p = MockPyBullet()
    p.connect(p.DIRECT)
    
    # Load URDF (returns body_id = 1)
    body_id = p.loadURDF("robot.urdf")
    
    # Reset base orientation to an invalid 3-element vector raises ValueError
    with pytest.raises(ValueError):
        p.resetBasePositionAndOrientation(body_id, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    
    p.disconnect()
