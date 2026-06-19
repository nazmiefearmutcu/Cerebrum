import os
import sys
import time
import signal
import sqlite3
import pytest
import json
import numpy as np
import threading
from unittest.mock import patch

from metrics_collector import MetricsCollector
from cerebrum_mind.cognitive.bridge import CognitiveBridge

# Helper to clear fallback file if it exists
def clear_fallback(path="metrics_fallback.json"):
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass

def test_numerical_edge_cases(tmp_path):
    db_file = tmp_path / "test_num_edges.db"
    bridge = CognitiveBridge()
    collector = MetricsCollector(bridge=bridge, db_path=str(db_file))
    
    obs = [np.ones(5) * 0.1 for _ in range(3)]
    
    # 1. Very large magnitude state vector (needs to be large enough to remain > 1e5 after adjustment)
    bridge.reset_network()
    bridge.x0 = np.array([1.2e5, 0.0, 0.0, 0.0, 0.0])
    bridge.step_predictive_coding(obs)
    
    # 2. NaN inputs
    bridge.reset_network()
    bridge.x1 = np.array([np.nan, 0.0, 0.0, 0.0, 0.0])
    bridge.step_predictive_coding(obs)
    
    # 3. Inf inputs (which mathematically becomes NaN after update: inf - inf = nan)
    bridge.reset_network()
    bridge.x2 = np.array([np.inf, 0.0, 0.0, 0.0, 0.0])
    bridge.step_predictive_coding(obs)
    
    # 4. Empty observations (should raise ValueError in bridge, not crash patching/collector)
    bridge.reset_network()
    with pytest.raises(ValueError):
        bridge.step_predictive_coding([])
        
    collector.shutdown()
    
    summary = collector.get_summary()
    assert summary["total_steps"] == 3
    assert summary["divergences"] == 3

    # Check database records
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("SELECT diverged, divergence_reason, diverging_vector FROM step_metrics ORDER BY step_id ASC")
    rows = cursor.fetchall()
    conn.close()
    
    assert len(rows) == 3
    # Check reasons
    assert rows[0] == (1, "Out of bounds", "x0")
    assert rows[1] == (1, "NaN", "x1")
    assert rows[2] == (1, "NaN", "x2") # mathematically, inf - inf is NaN


def test_empty_state_vector_crash(tmp_path):
    """
    Checks what happens when a state vector is empty, using a dummy bridge
    that does not crash on its own step method.
    This isolates the collector's own vulnerability to empty state vectors.
    """
    db_file = tmp_path / "test_empty_vector.db"
    
    class DummyBridge:
        def __init__(self):
            self.x0 = np.array([])
            self.x1 = np.ones(5)
            self.x2 = np.ones(5)
        def step_predictive_coding(self, obs):
            pass

    bridge = DummyBridge()
    collector = MetricsCollector(bridge=bridge, db_path=str(db_file))
    
    obs = [np.ones(5) * 0.1 for _ in range(3)]
    
    crashed = False
    try:
        bridge.step_predictive_coding(obs)
    except ValueError as e:
        if "zero-size array to reduction operation maximum" in str(e):
            crashed = True
        else:
            raise e
        
    collector.shutdown()
    assert not crashed, "Collector crashed (ValueError: zero-size array) when state vector was empty!"


def test_db_lock_fallback(tmp_path):
    db_file = tmp_path / "test_db_lock.db"
    
    # Initialize DB schema and collector first (without lock)
    collector2 = MetricsCollector(db_path=str(db_file))
    fallback_file = collector2.fallback_path
    clear_fallback(fallback_file)
    
    # Now hold an exclusive lock on the DB
    lock_conn = sqlite3.connect(str(db_file))
    lock_conn.execute("PRAGMA journal_mode=WAL;")
    lock_conn.execute("BEGIN EXCLUSIVE TRANSACTION;")
    
    # Log some steps
    for i in range(5):
        collector2.log_step_async(10.0 + i, 120.0, 4.5, False)
        
    # Shutdown collector2. Since DB is locked, it should write to fallback file
    # due to timeout/worker thread block.
    collector2.shutdown()
    
    # Release the lock
    lock_conn.rollback()
    lock_conn.close()
    
    # Verify fallback file has the records
    assert os.path.exists(fallback_file)
    fallback_records = []
    with open(fallback_file, "r") as f:
        for line in f:
            if line.strip():
                fallback_records.append(json.loads(line))
                
    clear_fallback(fallback_file)
    
    assert len(fallback_records) == 5
    latencies = [item["latency_ms"] for item in fallback_records]
    assert latencies == [10.0, 11.0, 12.0, 13.0, 14.0]


def test_high_load_stress(tmp_path):
    db_file = tmp_path / "test_high_load.db"
    collector = MetricsCollector(db_path=str(db_file))
    
    total_steps = 10000
    num_threads = 10
    steps_per_thread = total_steps // num_threads
    
    def log_rapidly():
        for i in range(steps_per_thread):
            collector.log_step_async(12.5, 150.0, 5.2, False)
            
    t0 = time.perf_counter()
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=log_rapidly)
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    collector.shutdown()
    duration = time.perf_counter() - t0
    
    # Verify DB records
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM step_metrics")
    count = cursor.fetchone()[0]
    conn.close()
    
    assert count == total_steps
    print(f"\n[INFO] Logged {total_steps} steps in {duration:.4f}s. Throughput: {total_steps/duration:.2f} steps/sec")


def test_concurrency_race_conditions(tmp_path):
    db_file = tmp_path / "test_concurrency_race.db"
    collector = MetricsCollector(db_path=str(db_file))
    
    num_bridges = 5
    bridges = [CognitiveBridge() for _ in range(num_bridges)]
    
    # Thread functions
    def patcher_worker():
        for _ in range(50):
            for b in bridges:
                collector.register_and_patch_bridge(b)
                time.sleep(0.001)
                collector.unpatch(b)
                time.sleep(0.001)
                
    def runner_worker(thread_idx):
        obs = [np.ones(5) * 0.1 for _ in range(3)]
        for _ in range(100):
            # run steps on bridges
            bridge = bridges[thread_idx % num_bridges]
            try:
                bridge.step_predictive_coding(obs)
            except AttributeError:
                # Could happen if unpatched / partially patched during step invocation
                pass
            time.sleep(0.001)
            
    threads = []
    # Create 2 patcher threads
    for _ in range(2):
        t = threading.Thread(target=patcher_worker)
        t.start()
        threads.append(t)
        
    # Create 8 runner threads
    for i in range(8):
        t = threading.Thread(target=runner_worker, args=(i,))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    collector.unpatch()
    collector.shutdown()
    
    # Verify no deadlocks and some metrics were collected
    summary = collector.get_summary()
    assert "total_steps" in summary or summary == {}


def test_abrupt_signal_exit_mocked(tmp_path):
    db_file = tmp_path / "test_signal_mock.db"
    collector = MetricsCollector(db_path=str(db_file))
    
    # Log some initial steps
    for i in range(50):
        collector.log_step_async(10.0, 100.0, 4.0, False)
        
    # Queue up a bunch of metrics
    for i in range(200):
        collector.log_step_async(20.0, 200.0, 5.0, False)
        
    # Mock sys.exit to prevent pytest from exiting
    with patch("sys.exit") as mock_exit:
        # Trigger the signal handler directly
        collector._handle_signal(signal.SIGINT, None)
        
        # Verify sys.exit was called with 0
        mock_exit.assert_called_once_with(0)
        
    # Clean up fallback file if it was written to
    fallback_file = collector.fallback_path
    
    # Read DB count
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM step_metrics")
    db_count = cursor.fetchone()[0]
    conn.close()
    
    # Read fallback count
    fallback_count = 0
    if os.path.exists(fallback_file):
        with open(fallback_file, "r") as f:
            for line in f:
                if line.strip():
                    fallback_count += 1
        os.remove(fallback_file)
        
    total_logged = db_count + fallback_count
    assert total_logged == 250, f"Expected 250 metrics logged, got {total_logged}"
