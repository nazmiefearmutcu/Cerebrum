"""
Continuous Stress & Memory Footprint Test Suite for Cerebrum-Mind.
"""

import pytest
import os
import time
import numpy as np
import psutil
from cerebrum.config import CerebrumConfig
from cerebrum.unified import CerebrumNet
from cerebrum.types import Exogenous

def test_cerebrum_sustained_stress_load():
    """Verify that Cerebrum-Mind maintains stable memory and latency over continuous runs."""
    cfg = CerebrumConfig(dims=(4, 8), n_settle=6, seed=42)
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg)
    process = psutil.Process(os.getpid())
    
    # Measure baseline memory
    baseline_mem = process.memory_info().rss / (1024 * 1024)
    peak_mem_growth = 0.0
    
    steps = 1000
    latencies = []
    
    # Stream high-frequency computational steps
    for step in range(steps):
        t0 = time.perf_counter()
        obs = [np.ones(4) * 0.5 for _ in range(2)]
        action = Exogenous(np.array([0.05, 0.05]))
        
        # Step network
        net.step(obs, action=action, reward=1.0)
        
        latencies.append((time.perf_counter() - t0) * 1000.0)
        
        # Track memory growth from baseline dynamically
        current_mem = process.memory_info().rss / (1024 * 1024)
        current_growth = current_mem - baseline_mem
        if current_growth > peak_mem_growth:
            peak_mem_growth = current_growth
        
    mean_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)
    max_latency = np.max(latencies)
    
    print(f"\n--- Cerebrum Stress Metrics ---")
    print(f"Peak Memory Growth: {peak_mem_growth:.3f} MB")
    print(f"Mean Latency: {mean_latency:.3f} ms")
    print(f"P99 Latency: {p99_latency:.3f} ms")
    print(f"Max Latency: {max_latency:.3f} ms")
    
    # Assert stable memory usage and latencies (strictly under limits)
    assert peak_mem_growth <= 50.0, f"Peak memory growth {peak_mem_growth:.3f} MB exceeded 50.0 MB limit"
    assert p99_latency < 50.0, f"P99 latency {p99_latency:.3f} ms exceeded 50.0 ms limit"
    assert max_latency < 100.0, f"Max latency {max_latency:.3f} ms exceeded 100.0 ms limit"

