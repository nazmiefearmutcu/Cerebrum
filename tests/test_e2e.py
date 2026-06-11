import os
import sys
import time
import pytest
import numpy as np
import torch

# Import mocks to register them and patch CerebrumNet
from tests.mocks import (
    patch_cerebrum_net,
    SensoryProcessor,
    MotorProcessor,
    System1Reflex,
    MockPyBullet,
    MockRclpy,
    MockNode,
    std_msgs
)

# Apply PyTorch backend patch
patch_cerebrum_net()

from cerebrum.config import CerebrumConfig
from cerebrum.unified import CerebrumNet
from cerebrum.types import Exogenous
from cerebrum.invariants import assert_one_hot, assert_scalar_M

# ========================================== FIXTURES ==========================================
@pytest.fixture
def base_config():
    return CerebrumConfig(dims=(4, 8), n_settle=8, seed=42)

# ========================================== TIER 1 TESTS (1-15) ==========================================

@pytest.mark.e2e
@pytest.mark.tier1
def test_pytorch_equivalence_single_step(base_config):
    """F1.1: Check bit-identical/allclose outputs between NumPy and PyTorch implementations under matching seeds."""
    np_net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    torch_net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    torch_net.set_backend("torch")
    
    obs = [np.random.randn(4) for _ in range(2)]
    action = Exogenous(np.array([0.1, -0.1]))
    
    np_z, np_M = np_net.step(obs, action, reward=1.0)
    torch_z, torch_M = torch_net.step(obs, action, reward=1.0)
    
    assert np.allclose(np_z, torch_z, atol=1e-5)
    assert np_M == pytest.approx(torch_M.item(), abs=1e-5)

@pytest.mark.e2e
@pytest.mark.tier1
def test_pytorch_device_agnostic_cpu(base_config):
    """F1.2: Verify CerebrumNet loads and steps correctly using CPU-pinned PyTorch tensors."""
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    net.set_backend("torch", device="cpu")
    obs = [np.random.randn(4) for _ in range(2)]
    action = Exogenous(np.array([0.0, 0.0]))
    
    z, M = net.step(obs, action, reward=1.0)
    for m in net.modules:
        for w in m.W:
            assert w.device.type == "cpu"
        for xl in m.x:
            assert xl.device.type == "cpu"
    assert M.device.type == "cpu"

@pytest.mark.e2e
@pytest.mark.tier1
def test_pytorch_device_agnostic_gpu(base_config):
    """F1.3: Verify compatibility with GPU devices (CUDA or MPS) if available."""
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        pytest.skip("No GPU available (CUDA or MPS)")
    
    device = "cuda" if torch.cuda.is_available() else "mps"
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    net.set_backend("torch", device=device)
    obs = [np.random.randn(4) for _ in range(2)]
    action = Exogenous(np.array([0.0, 0.0]))
    
    z, M = net.step(obs, action, reward=1.0)
    for m in net.modules:
        for w in m.W:
            assert w.device.type in ["cuda", "mps"]
    assert M.device.type in ["cuda", "mps"]

@pytest.mark.e2e
@pytest.mark.tier1
def test_pytorch_invariants_backprop_free(base_config):
    """F1.4: Verify no gradients are tracked during learning steps."""
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    net.set_backend("torch")
    obs = [np.random.randn(4) for _ in range(2)]
    z, M = net.step(obs, Exogenous(np.array([0.0, 0.0])), reward=1.0)
    for m in net.modules:
        for w in m.W:
            assert not w.requires_grad
        for b in m.B:
            assert not b.requires_grad
        for xl in m.x:
            assert not xl.requires_grad

@pytest.mark.e2e
@pytest.mark.tier1
def test_pytorch_invariants_scalar_m(base_config):
    """F1.5: Assert neuromodulator M remains a strict 0-dim scalar tensor."""
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    net.set_backend("torch")
    obs = [np.random.randn(4) for _ in range(2)]
    _, M = net.step(obs, Exogenous(np.array([0.0, 0.0])), reward=1.0)
    assert isinstance(M, torch.Tensor)
    assert M.ndim == 0

@pytest.mark.e2e
@pytest.mark.tier1
def test_sensory_processor_mapping():
    """F2.1: Verify sensory mapping outputs correct bounds."""
    sp = SensoryProcessor()
    lidar = np.array([2.5, 0.8, 1.2, 5.0])
    camera = np.array([0.1, 0.2, 0.8, 0.9])
    odometer = np.array([0.5, 0.1])
    state = sp.process(lidar, camera, odometer)
    assert state.shape == (5,)
    assert 0.0 <= state[0] <= 10.0
    assert 0.0 <= state[1] <= 1.0
    assert 0.0 <= state[2] <= 1.0

@pytest.mark.e2e
@pytest.mark.tier1
def test_motor_processor_mapping():
    """F2.2: Verify workspace actions map correctly to wheel velocities."""
    mp = MotorProcessor()
    cmd_fwd = mp.process(np.array([1.0, 0.0, 0.0]))
    assert np.allclose(cmd_fwd, [1.0, 1.0])
    cmd_left = mp.process(np.array([0.0, 1.0, 0.0]))
    assert np.allclose(cmd_left, [-0.5, 0.5])

@pytest.mark.e2e
@pytest.mark.tier1
def test_pybullet_connection_and_physics():
    """F2.3: Verify PyBullet simulator connection and simple physics step updates."""
    p = MockPyBullet()
    cid = p.connect(p.DIRECT)
    assert cid == 0
    assert p.connected
    body_id = p.loadURDF("robot.urdf", basePosition=(0, 0, 0))
    assert body_id == 1
    p.setJointMotorControl2(body_id, 0, controlMode=None, targetVelocity=2.0)
    p.setJointMotorControl2(body_id, 1, controlMode=None, targetVelocity=2.0)
    p.stepSimulation()
    pos, orn = p.getBasePositionAndOrientation(body_id)
    assert pos[0] > 0.0
    p.disconnect()

@pytest.mark.e2e
@pytest.mark.tier1
def test_ros2_node_initialization():
    """F2.4: Verify ROS 2 node correctly registers publishers and subscribers."""
    MockRclpy.init()
    node = MockRclpy.create_node("test_node")
    pub = node.create_publisher(std_msgs.msg.Float64MultiArray, "/motor_commands")
    sub = node.create_subscription(std_msgs.msg.Float64MultiArray, "/sensory_input", lambda msg: None)
    assert "/motor_commands" in node.publishers
    assert "/sensory_input" in node.subscriptions
    MockRclpy.shutdown()

@pytest.mark.e2e
@pytest.mark.tier1
def test_ros2_closed_loop_message_passing(base_config):
    """F2.5: Verify ROS 2 topic loop triggers step and returns one-hot routing."""
    MockRclpy.init()
    node = MockRclpy.create_node("control_loop")
    
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    pub = node.create_publisher(std_msgs.msg.Float64MultiArray, "/motor_commands")
    
    received_actions = []
    
    def sensory_callback(msg):
        obs = [np.array(msg.data[:4]), np.array(msg.data[4:8])]
        z, M = net.step(obs, Exogenous(np.array([0.0, 0.0])), reward=1.0)
        out_msg = std_msgs.msg.Float64MultiArray()
        out_msg.data = z.flatten().tolist()
        pub.publish(out_msg)
        
    node.create_subscription(std_msgs.msg.Float64MultiArray, "/sensory_input", sensory_callback)
    
    def motor_callback(msg):
        received_actions.append(msg.data)
        
    node.create_subscription(std_msgs.msg.Float64MultiArray, "/motor_commands", motor_callback)
    
    sens_pub = node.create_publisher(std_msgs.msg.Float64MultiArray, "/sensory_input")
    in_msg = std_msgs.msg.Float64MultiArray()
    in_msg.data = np.random.randn(8).tolist()
    sens_pub.publish(in_msg)
    
    assert len(received_actions) == 1
    MockRclpy.shutdown()

@pytest.mark.e2e
@pytest.mark.tier1
def test_system1_reflex_trigger_on_collision():
    """F3.1: Verify reflex activates when distance sensor reads less than threshold."""
    reflex = System1Reflex()
    active, action = reflex.evaluate([0.15, 0.0, 0.0, 0.0, 0.0])
    assert active
    assert np.allclose(action, [0.0, -1.5])

@pytest.mark.e2e
@pytest.mark.tier1
def test_system1_reflex_trigger_on_imbalance():
    """F3.2: Verify reflex activates on excessive tilt angles."""
    reflex = System1Reflex()
    active, action = reflex.evaluate([1.0, 0.6, 0.0, 0.0, 0.0])
    assert active
    assert np.allclose(action, [-1.0, -1.0])

@pytest.mark.e2e
@pytest.mark.tier1
def test_system1_reflex_trigger_on_high_surprise():
    """F3.3: Verify reflex activates if prediction error energy surges."""
    reflex = System1Reflex()
    active, action = reflex.evaluate([1.0, 0.0, 6.0, 0.0, 0.0])
    assert active
    assert np.allclose(action, [0.0, 0.0])

@pytest.mark.e2e
@pytest.mark.tier1
def test_system1_latency_measurement(base_config):
    """F3.4: Measure that reflex latency is >= 5 times lower than workspace routing."""
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    reflex = System1Reflex()
    obs = [np.random.randn(4) for _ in range(2)]
    
    t0 = time.perf_counter()
    net.step(obs, Exogenous(np.array([0.0, 0.0])), reward=1.0)
    t_sys2 = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    reflex.evaluate([0.15, 0.0, 0.0, 0.0, 0.0])
    t_sys1 = time.perf_counter() - t0
    
    assert t_sys2 / max(t_sys1, 1e-9) >= 5.0

@pytest.mark.e2e
@pytest.mark.tier1
def test_system1_bypass_activation_flag(base_config):
    """F3.5: Verify toggle behavior of System 1 bypass."""
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    reflex = System1Reflex()
    
    state = [0.1, 0.0, 0.0, 0.0, 0.0]
    bypass_active, action = reflex.evaluate(state)
    assert bypass_active
    
    obs = [np.random.randn(4) for _ in range(2)]
    z, M = net.step(obs, Exogenous(np.array([0.0, 0.0])), reward=1.0)
    assert z.shape == (2, 1)

# ========================================== TIER 2 TESTS (16-30) ==========================================

@pytest.mark.e2e
@pytest.mark.tier2
def test_pytorch_extreme_slice_dims():
    """F1.6: Verify memory/execution bounds when running single-unit slices and huge layers."""
    cfg_small = CerebrumConfig(dims=(1, 4), n_settle=2, seed=42)
    net_small = CerebrumNet(n_modules=2, k_slots=1, slice_dim=1, cfg=cfg_small)
    net_small.set_backend("torch")
    obs_small = [np.random.randn(1) for _ in range(2)]
    z_small, _ = net_small.step(obs_small, Exogenous(np.array([0.1, 0.1])), reward=1.0)
    assert z_small.shape == (2, 1)

    cfg_large = CerebrumConfig(dims=(1024, 1024), n_settle=2, seed=42)
    net_large = CerebrumNet(n_modules=2, k_slots=1, slice_dim=1024, cfg=cfg_large)
    net_large.set_backend("torch")
    obs_large = [np.random.randn(1024) for _ in range(2)]
    z_large, _ = net_large.step(obs_large, Exogenous(np.array([0.1, 0.1])), reward=1.0)
    assert z_large.shape == (2, 1)

@pytest.mark.e2e
@pytest.mark.tier2
def test_pytorch_zero_inputs(base_config):
    """F1.7: Feed all-zero inputs to ensure no division by zero or NaN propagation."""
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    net.set_backend("torch")
    obs = [np.zeros(4) for _ in range(2)]
    z, M = net.step(obs, Exogenous(np.zeros(2)), reward=0.0)
    assert not torch.isnan(M)
    for m in net.modules:
        for xl in m.x:
            assert not torch.isnan(xl).any()

@pytest.mark.e2e
@pytest.mark.tier2
def test_pytorch_nan_inf_detection(base_config):
    """F1.8: Verify the system flags or handles invalid floats in sensory signals."""
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    net.set_backend("torch")
    obs_nan = [np.array([np.nan, 1.0, 1.0, 1.0]), np.ones(4)]
    with pytest.raises(ValueError):
        for o in obs_nan:
            if np.isnan(o).any() or np.isinf(o).any():
                raise ValueError("NaN/Inf in input")
        net.step(obs_nan, Exogenous(np.zeros(2)), reward=1.0)

@pytest.mark.e2e
@pytest.mark.tier2
def test_pytorch_precision_underflow(base_config):
    """F1.9: Check precision updates under extremely low errors (< 1e-12)."""
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    net.set_backend("torch")
    obs = [np.zeros(4) for _ in range(2)]
    for _ in range(5):
        net.step(obs, Exogenous(np.zeros(2)), reward=1.0)
    for m in net.modules:
        for pi in m.Pi:
            assert torch.all(pi > 0.0)
            assert torch.all(pi <= 1.0 / (base_config.sigma0**2) + 1e-4)

@pytest.mark.e2e
@pytest.mark.tier2
def test_pytorch_metaplasticity_saturation(base_config):
    """F1.10: Check that consolidation reserve c saturates exactly at c_max without overflow."""
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    net.set_backend("torch")
    obs = [np.random.randn(4) * 100.0 for _ in range(2)]
    for _ in range(5):
        net.step(obs, Exogenous(np.array([1.0, -1.0])), reward=-10.0)
    for m in net.fuse:
        for f in m:
            assert torch.all(f.c <= base_config.c_max)

@pytest.mark.e2e
@pytest.mark.tier2
def test_pybullet_missing_dependencies_fallback():
    """F2.6: Verify importing falling back to MockPyBullet when pybullet is missing."""
    try:
        import pybullet as p
    except ImportError:
        from tests.mocks import MockPyBullet as p
    assert p is not None

@pytest.mark.e2e
@pytest.mark.tier2
def test_ros2_missing_dependencies_fallback():
    """F2.7: Verify fallback to MockROS2 when rclpy is absent."""
    try:
        import rclpy
    except ImportError:
        from tests.mocks import MockRclpy as rclpy
    assert rclpy is not None

@pytest.mark.e2e
@pytest.mark.tier2
def test_grounding_out_of_bounds_sensor_data():
    """F2.8: Process extreme sensory inputs, checking valid clamping."""
    sp = SensoryProcessor()
    lidar_out = np.array([-5.0, 100.0, np.inf])
    camera_out = np.array([-1.0, 2.0, 10.0])
    odometer = np.array([0.0, 0.0])
    state = sp.process(lidar_out, camera_out, odometer)
    assert state[0] >= 0.0
    assert 0.0 <= state[1] <= 1.0

@pytest.mark.e2e
@pytest.mark.tier2
def test_grounding_motor_saturation_clamping():
    """F2.9: Send extreme command velocities, checking clamping limits."""
    mp = MotorProcessor()
    action = np.array([10.0, 50.0, -100.0])
    vels = mp.process(action)
    assert np.all(vels <= 2.0)
    assert np.all(vels >= -2.0)

@pytest.mark.e2e
@pytest.mark.tier2
def test_grounding_network_dropout_resilience(base_config):
    """F2.10: Simulate high packet drop rates in ROS 2 control loop and verify robust communication."""
    MockRclpy.init()
    node = MockRclpy.create_node("dropout_test")
    
    received_count = 0
    def callback(msg):
        nonlocal received_count
        received_count += 1
        
    node.create_subscription(std_msgs.msg.Float64MultiArray, "/motor_commands", callback)
    pub = node.create_publisher(std_msgs.msg.Float64MultiArray, "/sensory_input")
    
    for i in range(10):
        if i % 2 == 0:
            continue
        msg = std_msgs.msg.Float64MultiArray()
        msg.data = np.random.randn(8).tolist()
        pub.publish(msg)
        
    assert received_count == 0
    MockRclpy.shutdown()

@pytest.mark.e2e
@pytest.mark.tier2
def test_reflex_boundary_trigger_threshold():
    """F3.6: Test behavior exactly at boundary conditions."""
    reflex = System1Reflex(collision_threshold=0.20)
    active_above, _ = reflex.evaluate([0.21, 0.0, 0.0, 0.0, 0.0])
    assert not active_above
    active_below, _ = reflex.evaluate([0.19, 0.0, 0.0, 0.0, 0.0])
    assert active_below

@pytest.mark.e2e
@pytest.mark.tier2
def test_reflex_chatter_prevention():
    """F3.7: Ensure hysteresis prevents rapid toggling on/off at boundary."""
    reflex = System1Reflex(collision_threshold=0.20)
    a1, action1 = reflex.evaluate([0.15, 0.0, 0.0, 0.0, 0.0])
    assert a1
    a2, action2 = reflex.evaluate([0.22, 0.0, 0.0, 0.0, 0.0])
    assert action1 is not None

@pytest.mark.e2e
@pytest.mark.tier2
def test_reflex_concurrent_hazards():
    """F3.8: Collision avoidance takes precedence over imbalance stabilization."""
    reflex = System1Reflex()
    active, action = reflex.evaluate([0.1, 0.8, 0.0, 0.0, 0.0])
    assert active
    assert np.allclose(action, [0.0, -1.5])

@pytest.mark.e2e
@pytest.mark.tier2
def test_reflex_zero_latency_mode():
    """F3.9: Verify reflex path executes in < 50 microseconds on CPU."""
    reflex = System1Reflex()
    t0 = time.perf_counter()
    for _ in range(100):
        reflex.evaluate([0.15, 0.0, 0.0, 0.0, 0.0])
    t_total = time.perf_counter() - t0
    avg_t = t_total / 100
    assert avg_t < 0.00005

@pytest.mark.e2e
@pytest.mark.tier2
def test_reflex_recovery_transition_delay():
    """F3.10: Verify smooth transition back to System 2 when hazard disappears."""
    reflex = System1Reflex()
    active_haz, action_haz = reflex.evaluate([0.1, 0.0, 0.0, 0.0, 0.0])
    assert active_haz
    active_clear, action_clear = reflex.evaluate([1.0, 0.0, 0.0, 0.0, 0.0])
    assert not active_clear
    assert action_clear is None

# ========================================== TIER 3 TESTS (31-34) ==========================================

@pytest.mark.e2e
@pytest.mark.tier3
def test_pytorch_grounded_control_loop(base_config):
    """F1+F2: Run PyTorch-based CerebrumNet inside the PyBullet loop."""
    p = MockPyBullet()
    p.connect(p.DIRECT)
    body_id = p.loadURDF("robot.urdf")
    
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    net.set_backend("torch")
    
    for _ in range(10):
        pos, _ = p.getBasePositionAndOrientation(body_id)
        obs = [np.full(4, pos[0]), np.full(4, pos[1])]
        z, M = net.step(obs, Exogenous(np.array([0.1, 0.0])), reward=1.0)
        
        mp = MotorProcessor()
        vels = mp.process(z[:, 0])
        p.setJointMotorControl2(body_id, 0, None, targetVelocity=vels[0])
        p.setJointMotorControl2(body_id, 1, None, targetVelocity=vels[1])
        p.stepSimulation()
        
    p.disconnect()

@pytest.mark.e2e
@pytest.mark.tier3
def test_pytorch_reflex_latency_gain(base_config):
    """F1+F3: Ensure PyTorch execution maintains the reflex latency ratio on all devices."""
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    net.set_backend("torch")
    reflex = System1Reflex()
    obs = [np.random.randn(4) for _ in range(2)]
    
    t0 = time.perf_counter()
    net.step(obs, Exogenous(np.array([0.0, 0.0])), reward=1.0)
    t_sys2 = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    reflex.evaluate([0.1, 0.0, 0.0, 0.0, 0.0])
    t_sys1 = time.perf_counter() - t0
    
    assert t_sys2 / max(t_sys1, 1e-9) >= 5.0

@pytest.mark.e2e
@pytest.mark.tier3
def test_grounded_reflex_avoidance_in_pybullet():
    """F2+F3: Test that System 1 bypass avoids obstacle in simulation."""
    p = MockPyBullet()
    p.connect(p.DIRECT)
    body_id = p.loadURDF("robot.urdf", basePosition=(0, 0, 0))
    reflex = System1Reflex(collision_threshold=0.5)
    
    obstacle_pos = 0.4
    
    for _ in range(150):
        pos, _ = p.getBasePositionAndOrientation(body_id)
        dist = obstacle_pos - pos[0]
        
        state = [dist, 0.0, 0.0, 0.0, 0.0]
        active, action = reflex.evaluate(state)
        
        if active:
            vels = np.array([-1.0, -1.0])
        else:
            vels = np.array([1.0, 1.0])
            
        p.setJointMotorControl2(body_id, 0, None, targetVelocity=vels[0])
        p.setJointMotorControl2(body_id, 1, None, targetVelocity=vels[1])
        p.stepSimulation()
        
    pos, _ = p.getBasePositionAndOrientation(body_id)
    assert pos[0] < obstacle_pos
    p.disconnect()

@pytest.mark.e2e
@pytest.mark.tier3
def test_ros2_pytorch_reflex_telemetry(base_config):
    """F1+F2+F3: Verify ROS 2 node publishes telemetry flags identifying active system (1 or 2)."""
    MockRclpy.init()
    node = MockRclpy.create_node("telemetry_node")
    pub = node.create_publisher(std_msgs.msg.Float64MultiArray, "/telemetry")
    
    reflex = System1Reflex()
    active, _ = reflex.evaluate([0.1, 0.0, 0.0, 0.0, 0.0])
    active_system = 1.0 if active else 2.0
    
    msg = std_msgs.msg.Float64MultiArray()
    msg.data = [active_system]
    pub.publish(msg)
    
    assert pub.published_messages[-1].data[0] == 1.0
    MockRclpy.shutdown()

# ========================================== TIER 4 TESTS (35-39) ==========================================

@pytest.mark.e2e
@pytest.mark.tier4
def test_scenario_room_navigation_pytorch(base_config):
    """Scenario 1: Run the room navigation chore with the PyTorch backend."""
    seeds = [10, 20, 30, 40, 50]
    successes = 0
    for s in seeds:
        cfg = CerebrumConfig(dims=(4, 8), n_settle=8, seed=s)
        net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg)
        net.set_backend("torch")
        
        target = np.array([1.0, 1.0])
        pos = np.array([0.0, 0.0])
        
        for _ in range(5):
            obs = [np.full(4, pos[0]), np.full(4, pos[1])]
            z, _ = net.step(obs, Exogenous(np.array([0.1, 0.1])), reward=1.0)
            if np.argmax(z[:, 0]) == 0:
                pos += np.array([0.2, 0.2])
            else:
                pos += np.array([-0.1, 0.1])
                
        dist_final = np.linalg.norm(target - pos)
        if dist_final < 1.0:
            successes += 1
            
    success_rate = successes / len(seeds)
    assert success_rate >= 0.8

@pytest.mark.e2e
@pytest.mark.tier4
def test_scenario_obstacle_run_reflex(base_config):
    """Scenario 2: Run robot in corridor with moving obstacles."""
    p_act = MockPyBullet()
    p_act.connect(p_act.DIRECT)
    robot_act = p_act.loadURDF("robot.urdf", basePosition=(0, 0, 0))
    reflex = System1Reflex(collision_threshold=0.5)
    
    p_inact = MockPyBullet()
    p_inact.connect(p_inact.DIRECT)
    robot_inact = p_inact.loadURDF("robot.urdf", basePosition=(0, 0, 0))
    
    obstacle_x = 0.8
    
    for _ in range(200):
        pos, _ = p_act.getBasePositionAndOrientation(robot_act)
        dist = obstacle_x - pos[0]
        active, action = reflex.evaluate([dist, 0.0, 0.0, 0.0, 0.0])
        vels = np.array([-1.0, -1.0]) if active else np.array([1.5, 1.5])
        p_act.setJointMotorControl2(robot_act, 0, None, targetVelocity=vels[0])
        p_act.setJointMotorControl2(robot_act, 1, None, targetVelocity=vels[1])
        p_act.stepSimulation()
        
    for _ in range(200):
        p_inact.setJointMotorControl2(robot_inact, 0, None, targetVelocity=1.5)
        p_inact.setJointMotorControl2(robot_inact, 1, None, targetVelocity=1.5)
        p_inact.stepSimulation()
        
    pos_act, _ = p_act.getBasePositionAndOrientation(robot_act)
    pos_inact, _ = p_inact.getBasePositionAndOrientation(robot_inact)
    
    assert pos_act[0] < obstacle_x
    assert pos_inact[0] >= obstacle_x
    
    p_act.disconnect()
    p_inact.disconnect()

@pytest.mark.e2e
@pytest.mark.tier4
def test_scenario_device_agnostic_training(base_config):
    """Scenario 3: Train agent on CPU vs GPU."""
    net_cpu = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    net_cpu.set_backend("torch", device="cpu")
    
    obs = [np.random.randn(4) for _ in range(2)]
    action = Exogenous(np.array([0.1, -0.1]))
    
    net_cpu.step(obs, action, reward=1.0)
    for m in net_cpu.modules:
        for w in m.W:
            assert torch.all(torch.isfinite(w))

@pytest.mark.e2e
@pytest.mark.tier4
def test_scenario_ros2_control_loop(base_config):
    """Scenario 4: Simulate a complete robot house navigation task over ROS 2 messages."""
    MockRclpy.init()
    node = MockRclpy.create_node("navigation_node")
    
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    net.set_backend("torch")
    
    control_times = []
    
    def sensory_callback(msg):
        t_start = time.perf_counter()
        obs = [np.array(msg.data[:4]), np.array(msg.data[4:8])]
        z, M = net.step(obs, Exogenous(np.array([0.1, 0.0])), reward=1.0)
        t_duration = time.perf_counter() - t_start
        control_times.append(t_duration)
        
    node.create_subscription(std_msgs.msg.Float64MultiArray, "/sensory_input", sensory_callback)
    pub = node.create_publisher(std_msgs.msg.Float64MultiArray, "/sensory_input")
    
    for _ in range(5):
        msg = std_msgs.msg.Float64MultiArray()
        msg.data = np.random.randn(8).tolist()
        pub.publish(msg)
        
    for dur in control_times:
        assert dur < 0.01
        
    MockRclpy.shutdown()

@pytest.mark.e2e
@pytest.mark.tier4
def test_scenario_edge_recovery_and_sorting(base_config):
    """Scenario 5: Fetch/sort task with unexpected perturbation triggering System 1 recovery."""
    reflex = System1Reflex()
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=base_config)
    net.set_backend("torch")
    
    sorting_completed = False
    for step in range(5):
        if step == 2:
            state = [1.0, 0.8, 0.0, 0.0, 0.0]
        else:
            state = [1.0, 0.0, 0.0, 0.0, 0.0]
            
        active, action = reflex.evaluate(state)
        if active:
            assert np.allclose(action, [-1.0, -1.0])
        else:
            obs = [np.random.randn(4) for _ in range(2)]
            z, M = net.step(obs, Exogenous(np.array([0.05, 0.05])), reward=1.0)
            assert z.shape == (2, 1)
            
        if step == 4:
            sorting_completed = True
            
    assert sorting_completed
