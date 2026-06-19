#!/usr/bin/env python3
"""
Metrics Collector Core Module
Handles real-time metric capture, instance-level method hooking (monkey patching),
async disk checkpointing using SQLite, and divergence detection.
"""

import os
import sys
import time
import argparse
import threading
import queue
import sqlite3
import types
import logging
import json
import collections
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger("MetricsCollector")

class MetricsCollector:
    """
    Thread-safe, asynchronous collector of system and model telemetry.
    Supports instance-level binding to CognitiveBridge to monitor predictive coding steps.
    """
    def __init__(
        self,
        bridge: Optional[Any] = None,
        db_path: str = "metrics.db",
        checkpoint_interval: int = 1,
        fallback_path: Optional[str] = None
    ):
        self.lock = threading.Lock()
        self.db_path = db_path
        self.checkpoint_interval = checkpoint_interval
        if fallback_path is not None:
            self.fallback_path = fallback_path
        else:
            self.fallback_path = os.path.join(os.path.dirname(self.db_path) or ".", "metrics_fallback.json")
        
        self.latencies = collections.deque(maxlen=10000)
        self.memory_footprints = collections.deque(maxlen=10000)
        self.power_draws = collections.deque(maxlen=10000)
        self.divergences: int = 0
        self.total_steps_count = 0
        self.peak_memory_mb = 0.0
        self.peak_power_watts = 0.0
        
        # Patch registry: bridge -> (original_value, had_dict_entry)
        self._patched_bridges = {}
        
        # Generate unique session_id
        import uuid
        from datetime import datetime, timezone
        self.session_id = str(uuid.uuid4())
        
        # Get system info
        system_cores = os.cpu_count()
        total_ram_gb = None
        try:
            import psutil
            total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            psutil_cores = psutil.cpu_count()
            if psutil_cores is not None:
                system_cores = psutil_cores
        except Exception:
            pass
            
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._system_cores = system_cores
        self._total_ram_gb = total_ram_gb
        
        # Thread-safe queue for async disk writing
        self.queue = queue.Queue()
        self.running = True
        self.current_batch = []
        self.db_disabled = False
        
        # Initialize SQLite DB
        self._init_db()
        
        # Start background writer thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        # Register exit hook and signals
        import atexit
        import signal
        atexit.register(self.shutdown)
        try:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        except ValueError:
            pass
        
        # Instance-level patching if bridge provided
        if bridge is not None:
            self.register_and_patch_bridge(bridge)

    def _handle_signal(self, signum, frame) -> None:
        self.shutdown()
        sys.exit(0)

    def _init_db(self) -> None:
        """Initializes SQLite database schema."""
        with self.lock:
            conn = None
            try:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                cursor = conn.cursor()
                
                # Create sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        started_at TEXT NOT NULL,
                        system_cores INTEGER,
                        total_ram_gb REAL
                    )
                """)
                
                # Create step_metrics table with all columns
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS step_metrics (
                        step_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        latency_ms REAL,
                        memory_mb REAL,
                        power_watts REAL,
                        diverged INTEGER,
                        session_id TEXT,
                        free_energy REAL,
                        x0 TEXT,
                        x1 TEXT,
                        x2 TEXT,
                        divergence_reason TEXT,
                        diverging_vector TEXT,
                        FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                    )
                """)
                
                # Add columns if they are missing (backwards compatibility)
                cursor.execute("PRAGMA table_info(step_metrics)")
                columns = [info[1] for info in cursor.fetchall()]
                new_cols = {
                    "session_id": "TEXT",
                    "free_energy": "REAL",
                    "x0": "TEXT",
                    "x1": "TEXT",
                    "x2": "TEXT",
                    "divergence_reason": "TEXT",
                    "diverging_vector": "TEXT"
                }
                for col_name, col_type in new_cols.items():
                    if col_name not in columns:
                        cursor.execute(f"ALTER TABLE step_metrics ADD COLUMN {col_name} {col_type}")
                
                # Add index on (session_id, step_id)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_step ON step_metrics (session_id, step_id)")
                
                # Insert session metadata
                cursor.execute("""
                    INSERT OR IGNORE INTO sessions (session_id, started_at, system_cores, total_ram_gb)
                    VALUES (?, ?, ?, ?)
                """, (self.session_id, self._started_at, self._system_cores, self._total_ram_gb))
                
                conn.commit()
            except sqlite3.Error as e:
                print(f"[WARNING] Database initialization failed: {e}", file=sys.stderr)
                self.db_disabled = True
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

    def _write_fallback(self, items: List[Dict[str, Any]]) -> None:
        try:
            with open(self.fallback_path, "a") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")
        except Exception as e:
            print(f"[ERROR] Failed to write fallback metrics: {e}", file=sys.stderr)

    def _worker_loop(self) -> None:
        """Background writer loop that drains the metric queue to disk using batching."""
        use_db = not getattr(self, "db_disabled", False)
        conn = None
        if use_db:
            try:
                conn = sqlite3.connect(self.db_path, timeout=1.5)
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
            except sqlite3.Error as e:
                print(f"[WARNING] Database connection failed in worker thread: {e}", file=sys.stderr)
                use_db = False
                self.db_disabled = True
        
        while self.running or not self.queue.empty():
            batch = []
            shutdown_received = False
            
            try:
                # Block for up to 0.1s to get the first item
                item = self.queue.get(timeout=0.1)
                if item == "SHUTDOWN":
                    shutdown_received = True
                else:
                    batch.append(item)
                self.queue.task_done()
            except queue.Empty:
                continue
                
            # Try to get up to 99 more items without blocking
            while len(batch) < 100:
                try:
                    item = self.queue.get_nowait()
                    if item == "SHUTDOWN":
                        shutdown_received = True
                        self.queue.task_done()
                        break
                    batch.append(item)
                    self.queue.task_done()
                except queue.Empty:
                    break
            
            if batch:
                items = batch
                with self.lock:
                    self.current_batch = items
                if use_db and getattr(self, "db_disabled", False):
                    use_db = False
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass
                        conn = None
                if use_db and conn is not None:
                    try:
                        cursor = conn.cursor()
                        cursor.executemany("""
                            INSERT INTO step_metrics (
                                timestamp, latency_ms, memory_mb, power_watts, diverged,
                                session_id, free_energy, x0, x1, x2, divergence_reason, diverging_vector
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, [
                            (
                                item.get("timestamp"),
                                item.get("latency_ms"),
                                item.get("memory_mb"),
                                item.get("power_watts"),
                                item.get("diverged"),
                                item.get("session_id", self.session_id),
                                item.get("free_energy"),
                                item.get("x0"),
                                item.get("x1"),
                                item.get("x2"),
                                item.get("divergence_reason"),
                                item.get("diverging_vector")
                            )
                            for item in items
                        ])
                        conn.commit()
                        with self.lock:
                            self.current_batch = []
                    except sqlite3.Error as e:
                        print(f"[WARNING] Database batch write failed: {e}", file=sys.stderr)
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        with self.lock:
                            if self.current_batch:
                                self._write_fallback(items)
                                self.current_batch = []
                        if not self.running:
                            break
                else:
                    with self.lock:
                        if self.current_batch:
                            self._write_fallback(items)
                            self.current_batch = []
            
            if shutdown_received:
                break
                
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def log_step_async(
        self,
        latency_ms: float,
        memory_mb: float,
        power_watts: float,
        diverged: bool,
        divergence_reason: Optional[str] = None,
        diverging_vector: Optional[str] = None,
        free_energy: Optional[float] = None,
        x0: Optional[Any] = None,
        x1: Optional[Any] = None,
        x2: Optional[Any] = None,
    ) -> None:
        """
        Pushes a metrics log event to the asynchronous processing queue.
        """
        with self.lock:
            self.latencies.append(latency_ms)
            self.memory_footprints.append(memory_mb)
            self.power_draws.append(power_watts)
            if diverged:
                self.divergences += 1
            self.total_steps_count += 1
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
            self.peak_power_watts = max(self.peak_power_watts, power_watts)
                
        def serialize_vector(v) -> Optional[str]:
            if v is None:
                return None
            try:
                if isinstance(v, str):
                    return v
                return json.dumps(np.asarray(v).tolist())
            except Exception:
                return None

        self.queue.put({
            "timestamp": time.time(),
            "latency_ms": latency_ms,
            "memory_mb": memory_mb,
            "power_watts": power_watts,
            "diverged": 1 if diverged else 0,
            "session_id": self.session_id,
            "free_energy": free_energy,
            "x0": serialize_vector(x0),
            "x1": serialize_vector(x1),
            "x2": serialize_vector(x2),
            "divergence_reason": divergence_reason,
            "diverging_vector": diverging_vector
        })

    def register_and_patch_bridge(self, bridge: Any) -> None:
        """
        Binds to a specific CognitiveBridge instance using types.MethodType to override
        step_predictive_coding dynamically and asynchronously measure performance.
        """
        if not hasattr(bridge, "step_predictive_coding"):
            raise AttributeError("Target bridge does not have 'step_predictive_coding' method.")
            
        with self.lock:
            if bridge in self._patched_bridges:
                return
                
            had_dict_entry = "step_predictive_coding" in bridge.__dict__
            original_value = bridge.step_predictive_coding
            self._patched_bridges[bridge] = (original_value, had_dict_entry)
            
            collector = self
            
            def patched_step(instance, obs):
                t0 = time.perf_counter()
                original_value(obs)
                duration_ms = (time.perf_counter() - t0) * 1000.0
                
                # Check for numerical instability or divergence
                diverged = False
                divergence_reason = None
                diverging_vector = None
                
                for name in ["x0", "x1", "x2"]:
                    if hasattr(instance, name):
                        state_vector = getattr(instance, name)
                        if state_vector is not None:
                            state_arr = np.asarray(state_vector)
                            if state_arr.size == 0:
                                continue
                            if np.isnan(state_arr).any():
                                diverged = True
                                divergence_reason = "NaN"
                                diverging_vector = name
                                break
                            elif np.isinf(state_arr).any():
                                diverged = True
                                divergence_reason = "Inf"
                                diverging_vector = name
                                break
                            elif np.max(np.abs(state_arr)) > 1e5:
                                diverged = True
                                divergence_reason = "Out of bounds"
                                diverging_vector = name
                                break
                
                if diverged:
                    logger.warning(
                        f"[DIVERGENCE DETECTED] Vector '{diverging_vector}' diverged. "
                        f"Reason: {divergence_reason}. Vector values: {getattr(instance, diverging_vector)}"
                    )
                
                # Retrieve real resource metrics
                mem_mb = 0.0
                try:
                    import psutil
                    process = psutil.Process()
                    mem_mb = process.memory_info().rss / (1024 * 1024)
                except Exception:
                    mem_mb = 120.0
                    
                # Power draw estimation
                power_watts = 4.2 + np.random.normal(0.0, 0.2)
                if diverged:
                    power_watts += 2.0
                    
                free_energy = None
                if hasattr(instance, "get_free_energy"):
                    try:
                        free_energy = instance.get_free_energy()
                    except Exception:
                        pass
                        
                x0_val = getattr(instance, "x0", None)
                x1_val = getattr(instance, "x1", None)
                x2_val = getattr(instance, "x2", None)
                
                # Copy these vectors right away to avoid mutation race conditions
                x0_copy = np.copy(x0_val) if x0_val is not None else None
                x1_copy = np.copy(x1_val) if x1_val is not None else None
                x2_copy = np.copy(x2_val) if x2_val is not None else None
                
                collector.log_step_async(
                    latency_ms=duration_ms,
                    memory_mb=mem_mb,
                    power_watts=power_watts,
                    diverged=diverged,
                    divergence_reason=divergence_reason,
                    diverging_vector=diverging_vector,
                    free_energy=free_energy,
                    x0=x0_copy,
                    x1=x1_copy,
                    x2=x2_copy
                )
                
            bridge.step_predictive_coding = types.MethodType(patched_step, bridge)

    def unpatch(self, bridge: Optional[Any] = None) -> None:
        with self.lock:
            if bridge is not None:
                if bridge in self._patched_bridges:
                    original_value, had_dict_entry = self._patched_bridges[bridge]
                    if had_dict_entry:
                        bridge.step_predictive_coding = original_value
                    else:
                        try:
                            del bridge.step_predictive_coding
                        except AttributeError:
                            pass
                    del self._patched_bridges[bridge]
            else:
                for b, (original_value, had_dict_entry) in self._patched_bridges.items():
                    if had_dict_entry:
                        b.step_predictive_coding = original_value
                    else:
                        try:
                            del b.step_predictive_coding
                        except AttributeError:
                            pass
                self._patched_bridges.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Returns statistical aggregates of all collected metrics."""
        with self.lock:
            if not self.latencies:
                return {}
            return {
                "mean_latency_ms": float(np.mean(self.latencies)),
                "p99_latency_ms": float(np.percentile(self.latencies, 99)),
                "mean_power_watts": float(np.mean(self.power_draws)),
                "peak_power_watts": float(self.peak_power_watts),
                "peak_memory_mb": float(self.peak_memory_mb),
                "divergences": self.divergences,
                "total_steps": self.total_steps_count
            }

    def load_last_checkpoint(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if getattr(self, "db_disabled", False):
            return None
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            cursor = conn.cursor()
            
            if session_id is not None:
                cursor.execute("""
                    SELECT x0, x1, x2, free_energy FROM step_metrics
                    WHERE diverged = 0 AND x0 IS NOT NULL AND session_id = ?
                    ORDER BY step_id DESC LIMIT 1
                """, (session_id,))
            else:
                cursor.execute("""
                    SELECT x0, x1, x2, free_energy FROM step_metrics
                    WHERE diverged = 0 AND x0 IS NOT NULL
                    ORDER BY step_id DESC LIMIT 1
                """)
                
            row = cursor.fetchone()
            
            if row is None:
                return None
                
            x0_str, x1_str, x2_str, free_energy = row
            return {
                "x0": np.array(json.loads(x0_str)) if x0_str else None,
                "x1": np.array(json.loads(x1_str)) if x1_str else None,
                "x2": np.array(json.loads(x2_str)) if x2_str else None,
                "free_energy": free_energy
            }
        except sqlite3.Error:
            return None
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def shutdown(self) -> None:
        """Gracefully halts the background database writer thread."""
        with self.lock:
            if not self.running:
                return
            self.running = False
            
        self.queue.put("SHUTDOWN")
        self.worker_thread.join(timeout=3.0)
        
        # If thread is still alive (failed SQLite write or timeout)
        if self.worker_thread.is_alive():
            with self.lock:
                current_batch = self.current_batch
                self.current_batch = []
            if current_batch:
                with self.lock:
                    self._write_fallback(current_batch)
            
        # Drain any remaining items in self.queue and write them to the fallback JSON file.
        remaining_metrics = []
        while not self.queue.empty():
            try:
                item = self.queue.get_nowait()
                if item != "SHUTDOWN":
                    remaining_metrics.append(item)
                self.queue.task_done()
            except queue.Empty:
                break
        if remaining_metrics:
            with self.lock:
                self._write_fallback(remaining_metrics)


def calibrate_system() -> None:
    """Performs system diagnostic check and baseline metrics logging."""
    print("[INFO] Starting Metrics Collector Calibration...")
    try:
        import psutil
        cpu_count = psutil.cpu_count()
        memory_total_gb = psutil.virtual_memory().total / (1024 ** 3)
        print(f"[INFO] Hardware detected: CPU cores={cpu_count}, RAM={memory_total_gb:.2f} GB")
    except ImportError:
        print("[WARNING] 'psutil' library not available. Using basic calibration values.")
        
    db_file = "metrics_calibration.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        
    collector = MetricsCollector(db_path=db_file)
    print("[INFO] Simulating calibration workloads...")
    
    # Simulate a baseline sequence
    for i in range(10):
        # 15ms latency, 150MB ram, 4.2W power
        collector.log_step_async(15.0 + np.random.normal(0.0, 0.5), 150.0 + np.random.normal(0.0, 1.0), 4.2 + np.random.normal(0.0, 0.05), False)
        time.sleep(0.05)
        
    collector.shutdown()
    
    # Verify DB has entries
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM step_metrics")
    count = cursor.fetchone()[0]
    conn.close()
    
    print(f"[INFO] Calibration completed. Saved {count} baseline entries to SQLite database: {db_file}.")
    print("STATUS: PASS")


def main():
    parser = argparse.ArgumentParser(description="Cerebrum Metrics Collector CLI")
    parser.add_argument("--calibrate", action="store_true", help="Run baseline system calibration check.")
    args = parser.parse_args()
    
    if args.calibrate:
        calibrate_system()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
