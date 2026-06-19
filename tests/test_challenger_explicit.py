import os
import sqlite3
import json
import pytest
import numpy as np
from metrics_collector import MetricsCollector

def test_verify_empty_state_vector():
    class DummyBridge:
        def __init__(self):
            self.x0 = np.array([])
            self.x1 = np.ones(5)
            self.x2 = np.ones(5)
        def step_predictive_coding(self, obs):
            pass

    bridge = DummyBridge()
    db_file = "test_empty_state.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        
    collector = MetricsCollector(bridge=bridge, db_path=db_file)
    obs = [np.ones(5) * 0.1 for _ in range(3)]
    
    # This should not raise any ValueError or exception
    bridge.step_predictive_coding(obs)
    collector.shutdown()
    
    if os.path.exists(db_file):
        os.remove(db_file)

def test_verify_db_lock_fallback_on_shutdown():
    db_file = "test_db_lock_shutdown.db"
    fallback_file = "test_fallback_shutdown.json"
    
    for f in [db_file, fallback_file]:
        if os.path.exists(f):
            os.remove(f)
            
    # Initialize DB (so it exists)
    collector = MetricsCollector(db_path=db_file, fallback_path=fallback_file)
    collector.shutdown()
    
    # Re-instantiate a collector to log metrics while DB is locked
    collector = MetricsCollector(db_path=db_file, fallback_path=fallback_file)
    
    # Hold exclusive lock on the DB
    lock_conn = sqlite3.connect(db_file)
    lock_conn.execute("PRAGMA journal_mode=WAL;")
    lock_conn.execute("BEGIN EXCLUSIVE TRANSACTION;")
    
    # Log metrics
    collector.log_step_async(latency_ms=45.6, memory_mb=120.0, power_watts=5.0, diverged=False)
    
    # Shutdown while DB is locked. This should flush items to fallback file.
    collector.shutdown()
    
    lock_conn.rollback()
    lock_conn.close()
    
    # Check fallback file
    assert os.path.exists(fallback_file), "Fallback file was not created!"
    with open(fallback_file, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]
        
    assert len(records) == 1, f"Expected 1 record in fallback, got {len(records)}"
    assert records[0]["latency_ms"] == 45.6, f"Expected latency 45.6, got {records[0]['latency_ms']}"
    
    for f in [db_file, fallback_file]:
        if os.path.exists(f):
            os.remove(f)

def test_verify_startup_db_lock():
    db_file = "test_startup_lock.db"
    fallback_file = "test_startup_fallback.json"
    
    for f in [db_file, fallback_file]:
        if os.path.exists(f):
            os.remove(f)
            
    # Create the db file and lock it exclusively
    lock_conn = sqlite3.connect(db_file)
    lock_conn.execute("PRAGMA journal_mode=WAL;")
    lock_conn.execute("BEGIN EXCLUSIVE TRANSACTION;")
    
    # Instantiate MetricsCollector. Should not crash and should set db_disabled=True
    collector = MetricsCollector(db_path=db_file, fallback_path=fallback_file)
    
    assert collector.db_disabled == True, "Expected db_disabled to be True under startup lock!"
    
    # Log a step
    collector.log_step_async(latency_ms=99.9, memory_mb=200.0, power_watts=6.0, diverged=False)
    
    # Shutdown
    collector.shutdown()
    
    lock_conn.rollback()
    lock_conn.close()
    
    # Check that fallback file contains the logged metric
    assert os.path.exists(fallback_file), "Fallback file was not created for disabled DB!"
    with open(fallback_file, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]
        
    assert len(records) == 1, f"Expected 1 record in fallback, got {len(records)}"
    assert records[0]["latency_ms"] == 99.9, f"Expected latency 99.9, got {records[0]['latency_ms']}"
    
    for f in [db_file, fallback_file]:
        if os.path.exists(f):
            os.remove(f)
