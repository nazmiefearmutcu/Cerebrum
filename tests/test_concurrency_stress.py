import time
import pytest
import numpy as np
import torch
import threading
import random

from cerebrum.config import CerebrumConfig
from cerebrum.unified import CerebrumNet
from cerebrum.types import Exogenous
from cerebrum.grounding import (
    MotorProcessor,
    CerebrumROSNode,
    MockPublisher,
    std_msgs,
    System1Reflex
)

def test_concurrent_stress():
    cfg = CerebrumConfig(dims=(4, 8), n_settle=4, seed=42)
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg)
    
    # Simple reflex that triggers bypass for small distance
    class MockReflex(System1Reflex):
        def evaluate(self, state):
            # If distance < 0.2, trigger reflex bypass
            dist = state.get("dist", 1.0)
            if dist < 0.2:
                return True, np.array([0.1, -0.1])
            return False, None

    reflex = MockReflex()
    node = CerebrumROSNode(net, node_name="stress_test_node", reflex=reflex)
    
    errors = []
    stop_event = threading.Event()
    
    def sensory_worker():
        while not stop_event.is_set():
            try:
                msg = std_msgs.msg.Float64MultiArray()
                # 50% chance of triggering reflex bypass (first element < 0.2)
                dist = random.uniform(0.01, 1.0)
                # length 8 sensory input
                msg.data = [dist, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                node.sensory_callback(msg)
                time.sleep(random.uniform(0.001, 0.01))
            except Exception as e:
                errors.append(f"Sensory callback error: {e}")
                
    def reward_worker():
        while not stop_event.is_set():
            try:
                msg = std_msgs.msg.Float64MultiArray()
                msg.data = [random.uniform(-1.0, 1.0)]
                node.reward_callback(msg)
                time.sleep(random.uniform(0.001, 0.01))
            except Exception as e:
                errors.append(f"Reward callback error: {e}")

    # Start 10 sensory workers and 10 reward workers
    threads = []
    for _ in range(10):
        threads.append(threading.Thread(target=sensory_worker))
        threads.append(threading.Thread(target=reward_worker))
        
    for t in threads:
        t.start()
        
    # Run for 2 seconds
    time.sleep(2.0)
    stop_event.set()
    
    for t in threads:
        t.join()
        
    assert len(errors) == 0, f"Thread safety stress test raised errors: {errors}"
