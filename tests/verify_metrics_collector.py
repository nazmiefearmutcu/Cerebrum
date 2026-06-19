import os
import sys
import time
import sqlite3
import shutil
import json
import numpy as np
import multiprocessing
from unittest.mock import patch

# Ensure the parent directory is in the path so we can import metrics_collector and cerebrum_mind
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from metrics_collector import MetricsCollector
from cerebrum_mind.cognitive.bridge import CognitiveBridge

def run_state_recovery_tests():
    print("\n--- Running State Recovery Tests ---")
    db_file = "test_state_recovery.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        
    try:
        # Scenario A & B: Interleaved diverged and non-diverged steps in a single session
        collector = MetricsCollector(db_path=db_file)
        session_id = collector.session_id
        
        # Log Step 1: non-diverged
        collector.log_step_async(
            latency_ms=10.0, memory_mb=120.0, power_watts=4.5, diverged=False,
            free_energy=0.5, x0=[1.0]*5, x1=[1.1]*5, x2=[1.2]*5
        )
        # Log Step 2: diverged
        collector.log_step_async(
            latency_ms=12.0, memory_mb=125.0, power_watts=6.5, diverged=True,
            divergence_reason="NaN", diverging_vector="x1",
            free_energy=10.0, x0=[2.0]*5, x1=[2.1]*5, x2=[2.2]*5
        )
        # Log Step 3: non-diverged
        collector.log_step_async(
            latency_ms=11.0, memory_mb=122.0, power_watts=4.6, diverged=False,
            free_energy=0.2, x0=[3.0]*5, x1=[3.1]*5, x2=[3.2]*5
        )
        # Log Step 4: diverged
        collector.log_step_async(
            latency_ms=14.0, memory_mb=128.0, power_watts=6.8, diverged=True,
            divergence_reason="Inf", diverging_vector="x2",
            free_energy=20.0, x0=[4.0]*5, x1=[4.1]*5, x2=[4.2]*5
        )
        
        collector.shutdown()
        
        # Load last checkpoint for session_id
        collector2 = MetricsCollector(db_path=db_file)
        checkpoint = collector2.load_last_checkpoint(session_id=session_id)
        
        assert checkpoint is not None, "Checkpoint should not be None"
        assert np.allclose(checkpoint["x0"], [3.0]*5), f"Expected x0 to be [3.0]*5, got {checkpoint['x0']}"
        assert np.allclose(checkpoint["x1"], [3.1]*5), f"Expected x1 to be [3.1]*5, got {checkpoint['x1']}"
        assert np.allclose(checkpoint["x2"], [3.2]*5), f"Expected x2 to be [3.2]*5, got {checkpoint['x2']}"
        assert np.isclose(checkpoint["free_energy"], 0.2), f"Expected free_energy to be 0.2, got {checkpoint['free_energy']}"
        print("[PASS] Interleaved diverged/non-diverged single session state recovery.")

        # Scenario C: Successive runs with distinct sessions
        # Create a second session (session B)
        collector3 = MetricsCollector(db_path=db_file)
        session_b = collector3.session_id
        
        collector3.log_step_async(
            latency_ms=15.0, memory_mb=130.0, power_watts=4.8, diverged=False,
            free_energy=0.1, x0=[10.0]*5, x1=[10.1]*5, x2=[10.2]*5
        )
        collector3.shutdown()
        
        # Query Session A using collector2
        checkpoint_a = collector2.load_last_checkpoint(session_id=session_id)
        assert checkpoint_a is not None
        assert np.allclose(checkpoint_a["x0"], [3.0]*5), "Session A checkpoint modified by Session B"
        
        # Query Session B using collector2
        checkpoint_b = collector2.load_last_checkpoint(session_id=session_b)
        assert checkpoint_b is not None
        assert np.allclose(checkpoint_b["x0"], [10.0]*5), f"Expected Session B checkpoint to be [10.0]*5, got {checkpoint_b['x0']}"
        
        # Query without session_id (should return overall latest, which is Session B)
        checkpoint_latest = collector2.load_last_checkpoint()
        assert checkpoint_latest is not None
        assert np.allclose(checkpoint_latest["x0"], [10.0]*5), "Global checkpoint should return the latest overall non-diverged step"
        print("[PASS] Successive runs with distinct sessions checkpoint query.")

        # Scenario D: Session with only diverged steps
        collector4 = MetricsCollector(db_path=db_file)
        session_c = collector4.session_id
        collector4.log_step_async(
            latency_ms=15.0, memory_mb=130.0, power_watts=6.8, diverged=True,
            divergence_reason="NaN", diverging_vector="x0",
            free_energy=5.0, x0=[100.0]*5, x1=[100.1]*5, x2=[100.2]*5
        )
        collector4.shutdown()
        
        checkpoint_c = collector2.load_last_checkpoint(session_id=session_c)
        assert checkpoint_c is None, f"Expected None for session with only diverged steps, got {checkpoint_c}"
        print("[PASS] Session with only diverged steps returns None.")
        
        collector2.shutdown()

    finally:
        if os.path.exists(db_file):
            os.remove(db_file)


def run_concurrency_worker(db_path, steps, barrier, finished_event):
    """Worker process that writes metrics to the shared DB."""
    # Initialize collector
    collector = MetricsCollector(db_path=db_path)
    
    # Wait for all processes to start
    barrier.wait()
    
    try:
        for i in range(steps):
            # Log some dummy steps
            collector.log_step_async(
                latency_ms=10.0 + np.random.normal(0.0, 1.0),
                memory_mb=150.0,
                power_watts=4.5,
                diverged=False,
                free_energy=0.5,
                x0=[float(i)] * 5,
                x1=[float(i)] * 5,
                x2=[float(i)] * 5
            )
            time.sleep(0.005 + np.random.uniform(0.0, 0.01)) # random delay
    finally:
        collector.shutdown()


def run_concurrency_reader(db_path, finished_event):
    """Reader process that continuously queries load_last_checkpoint."""
    collector = MetricsCollector(db_path=db_path)
    try:
        while not finished_event.is_set():
            try:
                checkpoint = collector.load_last_checkpoint()
                # Just verify it doesn't crash and returns valid data or None
                if checkpoint is not None:
                    assert "x0" in checkpoint
            except sqlite3.OperationalError as e:
                print(f"[FAIL] Reader got OperationalError: {e}", file=sys.stderr)
                raise
            time.sleep(0.01)
    finally:
        collector.shutdown()


def run_db_concurrency_tests():
    print("\n--- Running Database Concurrency/Locking Tests ---")
    db_file = "test_concurrency.db"
    fallback_file = "metrics_fallback.json"
    
    # Clean up previous runs
    for f in [db_file, db_file + "-wal", db_file + "-shm", fallback_file]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    # Number of concurrent writer processes
    num_processes = 6
    steps_per_process = 100
    
    # Use Multiprocessing Barrier and Event
    barrier = multiprocessing.Barrier(num_processes)
    finished_event = multiprocessing.Event()
    
    processes = []
    for _ in range(num_processes):
        p = multiprocessing.Process(
            target=run_concurrency_worker,
            args=(db_file, steps_per_process, barrier, finished_event)
        )
        processes.append(p)
        
    # Start reader process
    reader_p = multiprocessing.Process(
        target=run_concurrency_reader,
        args=(db_file, finished_event)
    )
    
    # Start all
    reader_p.start()
    for p in processes:
        p.start()
        
    # Wait for writers to complete
    for p in processes:
        p.join()
        
    # Signal reader to stop
    finished_event.set()
    reader_p.join()
    
    # Verify DB contents
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM step_metrics")
    count = cursor.fetchone()[0]
    conn.close()
    
    expected_count = num_processes * steps_per_process
    print(f"Logged steps: {count} / {expected_count}")
    
    # Check if fallback file was created
    fallback_exists = os.path.exists(fallback_file)
    
    # Assertions
    assert count == expected_count, f"Expected {expected_count} rows in DB, got {count}"
    assert not fallback_exists, "Fallback metrics file was created, indicating database lock errors!"
    print("[PASS] Concurrency tests passed. No lock errors occurred, and WAL mode/busy timeout worked.")
    
    # Cleanup
    for f in [db_file, db_file + "-wal", db_file + "-shm", fallback_file]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass


def run_psutil_fallback_tests():
    print("\n--- Running psutil Fallback Tests ---")
    db_file = "test_psutil_fallback.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        
    try:
        # First test WITH psutil (if available)
        print("Testing with psutil available...")
        collector = MetricsCollector(db_path=db_file)
        # Check system resource variables populated
        import psutil
        expected_cores = psutil.cpu_count() or os.cpu_count()
        assert collector._system_cores == expected_cores, f"Expected cores {expected_cores}, got {collector._system_cores}"
        assert collector._total_ram_gb is not None, "Expected total RAM to be populated"
        
        bridge = CognitiveBridge()
        collector.register_and_patch_bridge(bridge)
        
        obs = [np.ones(5) * 0.1 for _ in range(3)]
        bridge.step_predictive_coding(obs)
        collector.shutdown()
        
        # Verify db contains real RSS memory footprint (usually > 0.0 and not exactly 120.0 unless by chance)
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT memory_mb FROM step_metrics ORDER BY step_id DESC LIMIT 1")
        mem_val = cursor.fetchone()[0]
        conn.close()
        print(f"Logged memory with psutil: {mem_val} MB")
        
        # Clear database
        os.remove(db_file)
        
        # Now test WITHOUT psutil
        print("Testing with psutil mocked as missing...")
        with patch.dict('sys.modules', {'psutil': None}):
            collector_fallback = MetricsCollector(db_path=db_file)
            assert collector_fallback._system_cores == os.cpu_count(), f"Expected cores {os.cpu_count()}, got {collector_fallback._system_cores}"
            assert collector_fallback._total_ram_gb is None, "Expected total RAM to be None when psutil is missing"
            
            bridge_fallback = CognitiveBridge()
            collector_fallback.register_and_patch_bridge(bridge_fallback)
            bridge_fallback.step_predictive_coding(obs)
            collector_fallback.shutdown()
            
            # Verify db contains exactly 120.0 MB
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT memory_mb FROM step_metrics ORDER BY step_id DESC LIMIT 1")
            mem_val_fallback = cursor.fetchone()[0]
            conn.close()
            assert mem_val_fallback == 120.0, f"Expected fallback memory 120.0, got {mem_val_fallback}"
            print("[PASS] psutil missing fallback logic works perfectly.")
            
    finally:
        if os.path.exists(db_file):
            os.remove(db_file)


def run_profiling_tests():
    print("\n--- Running System Resource & Performance Profiling ---")
    db_file = "test_profiling.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        
    try:
        collector = MetricsCollector(db_path=db_file)
        
        # Measure initial memory
        import psutil
        process = psutil.Process()
        mem_start = process.memory_info().rss / (1024 * 1024)
        
        t0 = time.perf_counter()
        
        num_steps = 10000
        print(f"Logging {num_steps} steps rapidly...")
        for i in range(num_steps):
            collector.log_step_async(
                latency_ms=1.5,
                memory_mb=150.0,
                power_watts=4.5,
                diverged=False,
                free_energy=0.5,
                x0=[float(i)] * 5,
                x1=[float(i)] * 5,
                x2=[float(i)] * 5
            )
            
        t_enqueue = time.perf_counter() - t0
        print(f"Enqueued {num_steps} steps in {t_enqueue:.4f} seconds ({num_steps/t_enqueue:.1f} steps/sec).")
        
        # Shutdown to wait for database writer to finish
        t1 = time.perf_counter()
        collector.shutdown()
        t_drain = time.perf_counter() - t1
        t_total = time.perf_counter() - t0
        
        print(f"Drained and shutdown writer in {t_drain:.4f} seconds.")
        print(f"Total time: {t_total:.4f} seconds ({num_steps/t_total:.1f} overall steps/sec).")
        
        # Measure final memory
        mem_end = process.memory_info().rss / (1024 * 1024)
        mem_diff = mem_end - mem_start
        print(f"Process Memory before profiling: {mem_start:.2f} MB")
        print(f"Process Memory after profiling: {mem_end:.2f} MB")
        print(f"Process Memory growth: {mem_diff:.2f} MB")
        
        # Verify database has all entries
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM step_metrics")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == num_steps, f"Expected {num_steps} in DB, got {count}"
        print("[PASS] Profiling run successful.")
        
    finally:
        if os.path.exists(db_file):
            os.remove(db_file)


if __name__ == "__main__":
    run_state_recovery_tests()
    run_db_concurrency_tests()
    run_psutil_fallback_tests()
    run_profiling_tests()
    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
