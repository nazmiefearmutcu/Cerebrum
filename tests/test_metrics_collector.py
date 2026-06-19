import os
import sqlite3
import pytest
import numpy as np
import subprocess
import sys
import time

from metrics_collector import MetricsCollector
from cerebrum_mind.cognitive.bridge import CognitiveBridge

def test_metrics_collector_db_write(tmp_path):
    db_file = tmp_path / "test_metrics.db"
    collector = MetricsCollector(db_path=str(db_file))
    
    # Log metrics
    collector.log_step_async(10.0, 120.0, 4.5, False)
    collector.log_step_async(20.0, 130.0, 5.0, True)
    
    collector.shutdown()
    
    # Read DB contents
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("SELECT latency_ms, memory_mb, power_watts, diverged FROM step_metrics")
    rows = cursor.fetchall()
    conn.close()
    
    assert len(rows) == 2
    assert rows[0] == (10.0, 120.0, 4.5, 0)
    assert rows[1] == (20.0, 130.0, 5.0, 1)


def test_instance_level_monkey_patching(tmp_path):
    db_file = tmp_path / "test_patch.db"
    
    bridge_a = CognitiveBridge()
    bridge_b = CognitiveBridge()
    
    collector = MetricsCollector(bridge=bridge_a, db_path=str(db_file))
    
    # Run step on both bridges
    obs_a = [np.ones(5) * 0.1 for _ in range(3)]
    obs_b = [np.ones(5) * 0.2 for _ in range(3)]
    
    bridge_a.step_predictive_coding(obs_a)
    bridge_b.step_predictive_coding(obs_b)
    
    collector.shutdown()
    
    # bridge_a should trigger collector log, bridge_b should NOT
    summary = collector.get_summary()
    assert summary["total_steps"] == 1
    
    # Check that bridge_b was NOT patched (original method intact, no logging)
    assert "step_predictive_coding" not in bridge_b.__dict__
    assert "step_predictive_coding" in bridge_a.__dict__



def test_divergence_detection(tmp_path):
    db_file = tmp_path / "test_div.db"
    bridge = CognitiveBridge()
    collector = MetricsCollector(bridge=bridge, db_path=str(db_file))
    
    # Normal execution step
    obs = [np.ones(5) * 0.1 for _ in range(3)]
    bridge.step_predictive_coding(obs)
    
    # Induce NaN divergence
    bridge.x1[2] = np.nan
    bridge.step_predictive_coding(obs)
    
    # Induce inf divergence on another step
    bridge.x1[2] = 0.0 # reset NaN
    bridge.x2[0] = np.inf
    bridge.step_predictive_coding(obs)
    
    # Induce out-of-bounds divergence
    bridge.x2[0] = 1e6
    bridge.step_predictive_coding(obs)
    
    collector.shutdown()
    
    summary = collector.get_summary()
    assert summary["total_steps"] == 4
    # Three steps had divergence (NaN, inf, and 1e6)
    assert summary["divergences"] == 3


def test_calibrate_command():
    cmd = [sys.executable, "metrics_collector.py", "--calibrate"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "STATUS: PASS" in result.stdout
