import numpy as np
import torch
import pytest

from cerebrum.config import CerebrumConfig
from cerebrum.pc_core import PCAreas
from cerebrum.unified import CerebrumNet
from cerebrum.types import Exogenous
from cerebrum.hippocampus import Hippocampus
from cerebrum.grounding import SensoryProcessor, VLMAdapter, CerebrumROSNode
from benchmarks.baselines.er import run_continual_er, run_continual_der

def test_settling_stability():
    # Test L2 regularization and gradient clipping in PC areas
    cfg = CerebrumConfig(dims=(4, 8), pc_clip_value=0.5, pc_l2_decay=0.01)
    net = PCAreas(cfg)
    rng = np.random.default_rng(42)
    
    # Initialize extreme values to force clipping
    net.x[1] = torch.full_like(net.x[1], 100.0)
    net.eps[1] = torch.full_like(net.eps[1], 100.0)
    net.Pi[1] = torch.full_like(net.Pi[1], 10.0)
    
    # Run settle step
    net.settle_step(rng, T=0.01)
    
    # Check that x has decayed from 100.0 and hasn't exploded
    assert torch.all(torch.isfinite(net.x[1]))
    assert torch.mean(torch.abs(net.x[1])) < 100.0

def test_jit_acceleration():
    # Test JIT path in pc settling
    cfg = CerebrumConfig(dims=(4, 8), compile_modules=True)
    net = PCAreas(cfg)
    rng = np.random.default_rng(42)
    
    # Settle step with compile_modules=True
    net.settle_step(rng, T=0.01)
    assert torch.all(torch.isfinite(net.x[1]))

def test_experience_replay_baselines():
    # Test ER and DER++ baselines
    er_res = run_continual_er(seed=42, dim=4, per_task=2, passes=2, buffer_size=10)
    assert er_res["used_replay"] is True
    assert np.isfinite(er_res["forgetA"])
    
    der_res = run_continual_der(seed=42, dim=4, per_task=2, passes=2, buffer_size=10, alpha=0.5)
    assert der_res["used_der"] is True
    assert np.isfinite(der_res["forgetA"])

def test_sensor_fusion_and_randomization():
    # Test SensoryProcessor filtering and randomization
    proc = SensoryProcessor(alpha=0.5, randomize=True, noise_scale=0.1)
    
    # Run first processing pass
    lidar = np.array([2.0, 3.0, 4.0])
    cam = np.array([0.5, 0.5])
    odom = np.array([1.0, 0.0])
    
    state1 = proc.process(lidar, cam, odom)
    assert state1.shape == (5,)
    
    # Process identical inputs, and verify fusion reduces changes
    proc.randomize = False # disable noise to test filtering exactly
    proc.reset()
    state2 = proc.process(lidar, cam, odom)
    
    # Without noise, consecutive identical inputs result in identical outputs
    state3 = proc.process(lidar, cam, odom)
    np.testing.assert_allclose(state2, state3)

def test_hippocampus_episodic_memory():
    # Test Vector DB RAG Hippocampus
    hippo = Hippocampus(key_dim=4, capacity=3)
    
    # Store episodes
    hippo.write(np.array([1.0, 0.0, 0.0, 0.0]), "episode_a")
    hippo.write(np.array([0.0, 1.0, 0.0, 0.0]), "episode_b")
    hippo.write(np.array([0.0, 0.0, 1.0, 0.0]), "episode_c")
    
    # Retrieve closest match
    res = hippo.retrieve(np.array([0.9, 0.1, 0.0, 0.0]), k=1)
    assert len(res) == 1
    assert res[0]["value"] == "episode_a"
    assert res[0]["similarity"] > 0.9
    
    # Test LRU Eviction: writing a fourth episode must evict "episode_a" (oldest)
    hippo.write(np.array([0.0, 0.0, 0.0, 1.0]), "episode_d")
    
    assert hippo.size == 3
    # Verify episode_a is evicted (retrieve with a query matching episode_a is now different)
    res_evicted = hippo.retrieve(np.array([1.0, 0.0, 0.0, 0.0]), k=3)
    values = [r["value"] for r in res_evicted]
    assert "episode_a" not in values

def test_vlm_adapter():
    vlm = VLMAdapter()
    
    # NL instruction matching
    vec = vlm.bootstrap_command("Please find the red mug in the living room")
    assert vec[0] == 1.0 # mug_detected should be active
    
    vec_clean = vlm.bootstrap_command("Go clean the table")
    assert vec_clean[1] == 1.0 # table_dirty active
    
    # Visual processing mock
    vis = vlm.process_visual_scene(None)
    assert vis.shape == (5,)

def test_thread_decoupled_ros_node():
    cfg = CerebrumConfig(dims=(4, 8), n_settle=2, seed=42)
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg)
    
    # Instantiate ROS Node
    node = CerebrumROSNode(net=net)
    
    # Mock sensory message
    class MockMsg:
        def __init__(self, data):
            self.data = data
            
    # Send sensory input message
    msg = MockMsg(data=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.0])
    node.sensory_callback(msg)
    
    # Ensure immediate return (last_vels is default zeros)
    assert len(node.motor_pub.published_messages) == 1
    
    # Wait for the background thread to finish execution
    import time
    timeout = 1.0
    start_time = time.time()
    while node._thread_active and (time.time() - start_time) < timeout:
        time.sleep(0.01)
        
    assert node._thread_active is False
