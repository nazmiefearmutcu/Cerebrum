import os
import sqlite3
import pytest
import numpy as np
import subprocess
import sys
import time
import json

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


def test_explicit_unpatch(tmp_path):
    db_file = tmp_path / "test_unpatch.db"
    
    bridge_a = CognitiveBridge()
    bridge_b = CognitiveBridge()
    
    collector = MetricsCollector(db_path=str(db_file))
    
    # Patch both
    collector.register_and_patch_bridge(bridge_a)
    collector.register_and_patch_bridge(bridge_b)
    
    # Run step on both bridges
    obs_a = [np.ones(5) * 0.1 for _ in range(3)]
    obs_b = [np.ones(5) * 0.2 for _ in range(3)]
    
    bridge_a.step_predictive_coding(obs_a)
    bridge_b.step_predictive_coding(obs_b)
    
    # Check that both are logged
    summary = collector.get_summary()
    assert summary["total_steps"] == 2
    
    # Unpatch A
    collector.unpatch(bridge_a)
    
    # Running A should not log anymore
    bridge_a.step_predictive_coding(obs_a)
    summary = collector.get_summary()
    assert summary["total_steps"] == 2 # still 2
    
    # Running B should still log
    bridge_b.step_predictive_coding(obs_b)
    summary = collector.get_summary()
    assert summary["total_steps"] == 3 # incremented to 3
    
    # Unpatch all (B)
    collector.unpatch()
    
    # Running B should not log anymore
    bridge_b.step_predictive_coding(obs_b)
    summary = collector.get_summary()
    assert summary["total_steps"] == 3 # still 3
    
    collector.shutdown()


def test_patching_thread_safety(tmp_path):
    import concurrent.futures
    db_file = tmp_path / "test_threads.db"
    
    bridges = [CognitiveBridge() for _ in range(10)]
    collector = MetricsCollector(db_path=str(db_file))
    
    def worker(idx):
        # randomly patch/unpatch
        bridge = bridges[idx % len(bridges)]
        for _ in range(10):
            collector.register_and_patch_bridge(bridge)
            # simulate some work
            time.sleep(0.01)
            collector.unpatch(bridge)
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker, i) for i in range(20)]
        concurrent.futures.wait(futures)
        
    # verify clean state
    collector.unpatch()
    assert len(collector._patched_bridges) == 0
    collector.shutdown()


def test_state_checkpoint_and_recovery(tmp_path):
    db_file = tmp_path / "test_checkpoint.db"
    
    bridge = CognitiveBridge()
    collector = MetricsCollector(bridge=bridge, db_path=str(db_file))
    
    # Run a step to produce valid metrics/state
    obs = [np.ones(5) * 0.1 for _ in range(3)]
    bridge.step_predictive_coding(obs)
    
    # Shutdown to ensure DB write finishes
    session_id = collector.session_id
    collector.shutdown()
    
    # Now load checkpoint using a new collector instance
    collector2 = MetricsCollector(db_path=str(db_file))
    checkpoint = collector2.load_last_checkpoint(session_id=session_id)
    
    assert checkpoint is not None
    assert np.allclose(checkpoint["x0"], bridge.x0)
    assert np.allclose(checkpoint["x1"], bridge.x1)
    assert np.allclose(checkpoint["x2"], bridge.x2)
    assert np.isclose(checkpoint["free_energy"], bridge.get_free_energy())
    
    # Test loading last checkpoint without session_id
    checkpoint2 = collector2.load_last_checkpoint()
    assert checkpoint2 is not None
    assert np.allclose(checkpoint2["x0"], bridge.x0)
    
    collector2.shutdown()


def test_custom_fallback_path(tmp_path):
    """If a custom fallback_path is passed, verify it is used for writing fallback records."""
    db_file = tmp_path / "test_custom_fallback.db"
    custom_fallback = tmp_path / "my_custom_fallback.json"
    
    # Initialize with db locked from start to force fallback
    lock_conn = sqlite3.connect(str(db_file))
    lock_conn.execute("PRAGMA journal_mode=WAL;")
    lock_conn.execute("BEGIN EXCLUSIVE TRANSACTION;")
    
    collector = MetricsCollector(db_path=str(db_file), fallback_path=str(custom_fallback))
    collector.log_step_async(15.0, 140.0, 4.8, False)
    collector.shutdown()
    
    lock_conn.rollback()
    lock_conn.close()
    
    assert os.path.exists(str(custom_fallback))
    with open(custom_fallback, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 1
    assert records[0]["latency_ms"] == 15.0


def test_db_disabled_fallback_bypass(tmp_path):
    """If db_disabled is True, verify we bypass DB writing entirely and write directly to fallback."""
    db_file = tmp_path / "test_db_disabled.db"
    custom_fallback = tmp_path / "db_disabled_fallback.json"
    
    # We construct a collector but mock/force db_disabled = True
    collector = MetricsCollector(db_path=str(db_file), fallback_path=str(custom_fallback))
    collector.db_disabled = True
    
    # Log steps
    collector.log_step_async(25.0, 110.0, 4.0, False)
    collector.shutdown()
    
    # The DB file should either not exist or have no step_metrics rows (or not even session rows)
    # since we bypassed it. Let's verify fallback contains the record.
    assert os.path.exists(str(custom_fallback))
    with open(custom_fallback, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 1
    assert records[0]["latency_ms"] == 25.0


def test_cumulative_stats_peaks_and_steps(tmp_path):
    db_file = tmp_path / "test_cumulative.db"
    collector = MetricsCollector(db_path=str(db_file))
    
    # Force maxlen to be small to test deque eviction
    import collections
    collector.latencies = collections.deque(maxlen=3)
    collector.memory_footprints = collections.deque(maxlen=3)
    collector.power_draws = collections.deque(maxlen=3)
    
    # Log 5 steps with different values
    collector.log_step_async(10.0, 100.0, 5.0, False)
    collector.log_step_async(10.0, 200.0, 3.0, False) # peak memory
    collector.log_step_async(10.0, 150.0, 8.0, False) # peak power
    collector.log_step_async(10.0, 120.0, 4.0, False)
    collector.log_step_async(10.0, 110.0, 4.5, False)
    
    summary = collector.get_summary()
    
    assert len(collector.memory_footprints) == 3
    assert summary["peak_memory_mb"] == 200.0
    assert summary["peak_power_watts"] == 8.0
    assert summary["total_steps"] == 5
    
    collector.shutdown()


def test_sqlite_finally_blocks(tmp_path, monkeypatch):
    db_file = tmp_path / "test_finally.db"
    collector = MetricsCollector(db_path=str(db_file))
    
    original_connect = sqlite3.connect
    close_called = False
    
    class MockConn:
        def __init__(self, *args, **kwargs):
            self.real_conn = original_connect(*args, **kwargs)
        def execute(self, *args, **kwargs):
            return self.real_conn.execute(*args, **kwargs)
        def cursor(self, *args, **kwargs):
            class MockCursor:
                def execute(self, *args, **kwargs):
                    raise RuntimeError("Forced DB Error")
            return MockCursor()
        def close(self):
            nonlocal close_called
            close_called = True
            self.real_conn.close()
            
    monkeypatch.setattr(sqlite3, "connect", MockConn)
    
    try:
        collector.load_last_checkpoint()
    except RuntimeError:
        pass
        
    assert close_called, "Connection close was not called on load_last_checkpoint failure!"
    
    collector.shutdown()


def test_sqlite_startup_connection_failure_fallback(tmp_path, monkeypatch):
    """Verify that any sqlite3.Error during startup DB initialization is caught,
    warns to stderr, sets db_disabled=True, and falls back cleanly to json logging."""
    db_file = tmp_path / "corrupted_or_failing.db"
    
    original_connect = sqlite3.connect
    def mock_connect(*args, **kwargs):
        raise sqlite3.DatabaseError("Simulated connection/corruption error")
        
    monkeypatch.setattr(sqlite3, "connect", mock_connect)
    
    collector = MetricsCollector(db_path=str(db_file), fallback_path=str(tmp_path / "fallback.json"))
    
    assert collector.db_disabled is True
    
    collector.log_step_async(12.0, 150.0, 4.5, False)
    collector.shutdown()
    
    fallback_file = tmp_path / "fallback.json"
    assert os.path.exists(str(fallback_file))
    with open(fallback_file, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 1
    assert records[0]["latency_ms"] == 12.0


def test_metrics_collector_corrupt_db_file_fallback(tmp_path):
    """Verify that MetricsCollector can be initialized with a completely corrupt,
    non-SQLite file format database file and falls back cleanly to JSON logging."""
    db_file = tmp_path / "corrupt_junk.db"
    # Write corrupt non-SQLite junk
    with open(db_file, "wb") as f:
        f.write(b"This is not a SQLite database file! Random junk data 1234567890\n")
        
    fallback_file = tmp_path / "corrupt_fallback.json"
    
    from metrics_collector import MetricsCollector
    collector = MetricsCollector(db_path=str(db_file), fallback_path=str(fallback_file))
    
    assert collector.db_disabled is True
    
    # Try logging steps
    collector.log_step_async(15.0, 120.0, 3.8, False)
    collector.log_step_async(16.0, 121.0, 3.9, False)
    collector.shutdown()
    
    # Verify fallback has the steps logged correctly
    assert os.path.exists(str(fallback_file))
    with open(fallback_file, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 2
    assert records[0]["latency_ms"] == 15.0
    assert records[1]["latency_ms"] == 16.0


def test_metrics_collector_lock_contention_fallback(tmp_path, monkeypatch):
    """Verify that lock contention on the SQLite database causes MetricsCollector
    to fall back cleanly to JSON logging without crashing."""
    db_file = tmp_path / "locked_db.db"
    
    # Pre-create the database and lock it exclusively
    conn_lock = sqlite3.connect(str(db_file))
    conn_lock.execute("PRAGMA journal_mode=WAL;")
    conn_lock.execute("CREATE TABLE foo (bar INTEGER);")
    conn_lock.execute("BEGIN EXCLUSIVE TRANSACTION;")
    # keep conn_lock open so it holds the exclusive lock
    
    # Wrap sqlite3.connect to force a very low timeout so the test is fast
    original_connect = sqlite3.connect
    def fast_connect(*args, **kwargs):
        # Override timeout to 0.01 to trigger lock contention immediately
        kwargs["timeout"] = 0.01
        return original_connect(*args, **kwargs)
        
    monkeypatch.setattr(sqlite3, "connect", fast_connect)
    
    fallback_file = tmp_path / "lock_fallback.json"
    
    from metrics_collector import MetricsCollector
    collector = MetricsCollector(db_path=str(db_file), fallback_path=str(fallback_file))
    
    assert collector.db_disabled is True
    
    collector.log_step_async(20.0, 180.0, 5.0, True, divergence_reason="ContentionTest")
    collector.shutdown()
    
    # Close the lock connection
    conn_lock.close()
    
    # Verify fallback has the steps logged correctly
    assert os.path.exists(str(fallback_file))
    with open(fallback_file, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 1
    assert records[0]["latency_ms"] == 20.0
    assert records[0]["divergence_reason"] == "ContentionTest"




