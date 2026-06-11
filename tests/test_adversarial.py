import time
import pytest
import numpy as np
import torch
import threading
import sys

from cerebrum.config import CerebrumConfig
from cerebrum.unified import CerebrumNet
from cerebrum.types import Exogenous
from cerebrum.grounding import (
    MotorProcessor,
    CerebrumROSNode,
    MockPublisher,
    std_msgs,
    MockPyBullet
)

# ========================================== FIXTURES ==========================================
@pytest.fixture
def base_config():
    return CerebrumConfig(dims=(4, 8), n_settle=2, seed=42)

# ========================================== TIER 5 ADVERSARIAL TESTS ==========================

def test_nan_inf_sensory_propagation_vulnerability(base_config):
    """
    Adv-1: Verify NaN/Inf sensory inputs are sanitized and do not corrupt the network weights.
    """
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    node = CerebrumROSNode(net, node_name="nan_test_node")
    
    # Verify initial weights are finite
    for m in net.modules:
        for w in m.W:
            assert np.all(np.isfinite(w.cpu().numpy() if hasattr(w, 'cpu') else w))
            
    # Send sensory data with NaN values that bypasses the reflex
    msg = std_msgs.msg.Float64MultiArray()
    msg.data = [np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # Length 8, contains NaN
    
    # Execute sensory callback
    node.sensory_callback(msg)
    
    # Verify weights are not corrupted (contain no NaNs) after processing
    nan_found = False
    for m in net.modules:
        for w in m.W:
            w_np = w.cpu().numpy() if hasattr(w, 'cpu') else w
            if np.isnan(w_np).any():
                nan_found = True
                
    assert not nan_found, "NaNs propagated to network weights."


def test_motor_processor_linear_weight_dimension_crash():
    """
    Adv-2: Verify that initializing MotorProcessor in linear mode with high-dimensional 
    (3D+) W_motor does not crash during process() and handles alignment check.
    """
    # 3D W_motor of shape (2, 3, 4)
    W_motor_3d = np.ones((2, 3, 4))
    b_motor = np.zeros(2)
    mp = MotorProcessor(mode="linear", W_motor=W_motor_3d, b_motor=b_motor)
    
    # action_vector of length 3 matching W_motor.shape[1]
    action_vector = np.ones(3)
    
    # The call to process should complete without error, defaulting to zero velocities
    vels = mp.process(action_vector)
    assert np.allclose(vels, [0.0, 0.0])


def test_concurrent_execution_race_hazard(base_config):
    """
    Adv-3: Verify that concurrent calls to CerebrumNet.step or CerebrumROSNode.sensory_callback 
    from multiple threads are safe and do not raise exceptions due to thread safety.
    """
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    node = CerebrumROSNode(net, node_name="concurrent_test_node")
    
    # We will trigger sensory callbacks concurrently in multiple threads
    msg = std_msgs.msg.Float64MultiArray()
    msg.data = np.ones(8).tolist()
    
    errors = []
    def worker():
        try:
            for _ in range(5):
                node.sensory_callback(msg)
        except Exception as e:
            errors.append(e)
            
    threads = [threading.Thread(target=worker) for _ in range(5)]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    assert len(errors) == 0, f"Concurrency raised errors: {errors}"


def test_pybullet_double_simulation_divergence():
    """
    Adv-4: Verify that when PyBullet is simulated as available and steps successfully,
    stepSimulation() bypasses the manual Euler integration loop for self.bodies to avoid divergence.
    """
    import cerebrum
    
    # Mock pybullet as available
    orig_available = getattr(cerebrum, "PYBULLET_AVAILABLE", False)
    orig_real_p = getattr(cerebrum, "real_p", None)
    
    cerebrum.PYBULLET_AVAILABLE = True
    
    # Create a mock real_p module with a static base position
    class MockRealPyBullet:
        def connect(self, mode): return 0
        def disconnect(self): pass
        def loadURDF(self, path, basePosition=(0,0,0), baseOrientation=(0,0,0,1)):
            return 1
        def stepSimulation(self): pass
        def getBasePositionAndOrientation(self, body_id):
            return [10.0, 10.0, 10.0], [0.0, 0.0, 0.0, 1.0]
        def setJointMotorControl2(self, *args, **kwargs): pass
        
    cerebrum.real_p = MockRealPyBullet()
    
    p = MockPyBullet()
    p.connect(p.DIRECT)
    body_id = p.loadURDF("robot.urdf", basePosition=(0, 0, 0))
    
    # Set velocity so the mock updates
    p.setJointMotorControl2(body_id, 0, None, targetVelocity=2.0)
    p.setJointMotorControl2(body_id, 1, None, targetVelocity=2.0)
    
    # Step the simulation
    p.stepSimulation()
    
    # Retrieve the mock position stored inside p.bodies
    mock_pos = p.bodies[body_id]["pos"]
    
    # Bypassed manual integration should leave the mock_pos untouched at 0
    assert np.allclose(mock_pos, [0.0, 0.0, 0.0]), "Manual Euler integration was not bypassed."
    
    # Reset flag and module to clean up
    cerebrum.PYBULLET_AVAILABLE = orig_available
    cerebrum.real_p = orig_real_p


def test_ros_node_malformed_input_type_error(base_config):
    """
    Adv-5: Verify that passing non-numeric or malformed data to CerebrumROSNode.sensory_callback 
    does not crash and is handled safely.
    """
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    node = CerebrumROSNode(net, node_name="malformed_test_node")
    
    # Pass string data inside msg
    msg = std_msgs.msg.Float64MultiArray()
    msg.data = ["invalid", "data", "strings"]
    
    # This should not raise any exceptions
    node.sensory_callback(msg)
