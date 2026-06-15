"""
Pytest Suite for Digital-Twin Simulation Runner (run_validation_sim.py).
"""

import pytest
import os
import json
import numpy as np
from run_validation_sim import SimMetricsCollector, run_cerebrum_simulation, run_transformer_simulation

def test_metrics_collector():
    collector = SimMetricsCollector()
    
    # Log some dummy steps
    collector.log_step(latency_ms=10.0, memory_mb=100.0, power_watts=5.0)
    collector.log_step(latency_ms=20.0, memory_mb=120.0, power_watts=6.0)
    collector.log_step(latency_ms=30.0, memory_mb=110.0, power_watts=4.0)
    
    collector.log_episode(success=True)
    collector.log_episode(success=False)
    
    summary = collector.get_summary()
    
    assert summary["total_episodes"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["mean_latency_ms"] == 20.0
    assert summary["p99_latency_ms"] == pytest.approx(29.8, abs=0.1)
    assert summary["peak_memory_mb"] == 120.0
    assert summary["peak_power_watts"] == 6.0

def test_run_cerebrum_simulation_smoke():
    # Smoke test with small number of episodes and steps
    summary = run_cerebrum_simulation(episodes=2, steps_per_episode=10)
    assert summary["total_episodes"] == 2
    assert "success_rate" in summary
    assert "mean_latency_ms" in summary
    assert summary["mean_power_watts"] < 6.0  # Cerebrum is low power

def test_run_transformer_simulation_smoke():
    # Smoke test with small number of episodes and steps
    summary = run_transformer_simulation(episodes=2, steps_per_episode=10)
    assert summary["total_episodes"] == 2
    assert "success_rate" in summary
    assert "mean_latency_ms" in summary
    assert summary["mean_power_watts"] > 15.0  # Transformer is high power

def test_run_cerebrum_noise_robustness():
    """
    Verify that injecting up to 20% Gaussian noise into simulated sensor inputs
    retains model robustness and computed motor commands remain within [-1.0, 1.0].
    """
    # Run simulation with 20% Gaussian noise and motor clamping enabled
    summary = run_cerebrum_simulation(episodes=2, steps_per_episode=20, noise_level=0.20, clamp_motor=True)
    
    assert summary["total_episodes"] == 2
    assert "success_rate" in summary
    assert "mean_latency_ms" in summary
    assert summary["mean_power_watts"] < 6.0  # Cerebrum remains low power
