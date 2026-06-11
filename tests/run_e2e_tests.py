#!/usr/bin/env python3
import os
import sys
# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import json
import argparse
import numpy as np
import torch
import pytest

from cerebrum.config import CerebrumConfig
from cerebrum.unified import CerebrumNet
from cerebrum.types import Exogenous
from tests.mocks import patch_cerebrum_net, System1Reflex

# Apply PyTorch backend patch
patch_cerebrum_net()

def check_dependencies():
    deps = {"pybullet": False, "rclpy": False}
    try:
        import pybullet
        deps["pybullet"] = True
    except ImportError:
        pass
    try:
        import rclpy
        deps["rclpy"] = True
    except ImportError:
        pass
    return deps

class TestResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.total = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.total += 1
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1
            elif report.skipped:
                self.skipped += 1

def run_performance_profiling(device="cpu"):
    print("\n=== Running System 1 (Reflex) vs System 2 (Settling) Profiling ===")
    
    cfg = CerebrumConfig(dims=(4, 8), n_settle=8, seed=42)
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg)
    net.set_backend("torch", device=device)
    
    reflex = System1Reflex()
    
    # Warmup
    obs = [np.random.randn(4) for _ in range(2)]
    action = Exogenous(np.array([0.05, 0.05]))
    net.step(obs, action, reward=1.0)
    reflex.evaluate([0.5, 0.0, 0.0, 0.0, 0.0])
    
    # Generate 100 runs
    num_runs = 100
    sys2_times = []
    sys1_times = []
    
    sys2_ops = []
    
    for i in range(num_runs):
        # Generate some random observation and action
        obs = [np.random.randn(4) for _ in range(2)]
        act_val = np.random.uniform(-0.1, 0.1, size=2)
        action = Exogenous(act_val)
        
        # Profile System 2
        # reset counters
        net.counters.synaptic_ops = 0
        
        t0 = time.perf_counter_ns()
        net.step(obs, action, reward=1.0)
        t_sys2 = time.perf_counter_ns() - t0
        sys2_times.append(t_sys2 / 1000.0) # convert to microseconds
        sys2_ops.append(net.counters.synaptic_ops)
        
        # Profile System 1
        # Obstacle proximity varies
        dist = np.random.uniform(0.05, 1.0)
        state = [dist, 0.0, 0.0, 0.0, 0.0]
        
        t0 = time.perf_counter_ns()
        reflex.evaluate(state)
        t_sys1 = time.perf_counter_ns() - t0
        sys1_times.append(t_sys1 / 1000.0) # convert to microseconds
        
    mean_sys2_time = np.mean(sys2_times)
    mean_sys1_time = np.mean(sys1_times)
    mean_sys2_ops = np.mean(sys2_ops)
    latency_ratio = mean_sys2_time / max(mean_sys1_time, 1e-9)
    
    print(f"System 2 (Unified PC Settling):")
    print(f"  - Mean Latency: {mean_sys2_time:.3f} us")
    print(f"  - Mean Synaptic Ops: {mean_sys2_ops:.1f}")
    
    print(f"System 1 (Reflex Bypass):")
    print(f"  - Mean Latency: {mean_sys1_time:.3f} us")
    print(f"  - Mean Synaptic Ops: 0.0 (O(1) conditional check)")
    
    print(f"Latency Ratio (Sys2 / Sys1): {latency_ratio:.2f}x")
    ratio_passed = latency_ratio >= 5.0
    print(f"Reflex Bypass 5x Target: {'PASSED' if ratio_passed else 'FAILED'}")
    
    return {
        "system2": {
            "mean_latency_us": mean_sys2_time,
            "mean_synaptic_ops": mean_sys2_ops
        },
        "system1": {
            "mean_latency_us": mean_sys1_time,
            "mean_synaptic_ops": 0.0
        },
        "latency_ratio": latency_ratio,
        "ratio_passes_threshold": bool(ratio_passed)
    }

def main():
    parser = argparse.ArgumentParser(description="Cerebrum E2E Test Suite Runner & Profiler")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], help="Run specific test tier")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Force computation device")
    parser.add_argument("--report", type=str, help="Path to write the JSON test report")
    args = parser.parse_args()

    deps = check_dependencies()
    print("=== Cerebrum E2E Dependency Check ===")
    print(f"PyBullet: {'Available' if deps['pybullet'] else 'Missing (using mock)'}")
    print(f"ROS 2 (rclpy): {'Available' if deps['rclpy'] else 'Missing (using mock)'}")
    
    # Set PyTorch device context
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            
    print(f"Selected Device: {device}")
    
    # Formulate pytest arguments
    pytest_args = ["-p", "no:warnings", "-v"]
    if args.tier:
        pytest_args.extend(["-m", f"tier{args.tier}"])
    else:
        pytest_args.extend(["-m", "e2e"])
        
    pytest_args.append("tests/test_e2e.py")
    
    print(f"Running pytest with args: {pytest_args}")
    
    # Run pytest and collect results
    collector = TestResultCollector()
    exit_code = pytest.main(pytest_args, plugins=[collector])
    
    print(f"\n=== Test Execution Summary ===")
    print(f"Total E2E test cases: {collector.total}")
    print(f"Passed: {collector.passed}")
    print(f"Failed: {collector.failed}")
    print(f"Skipped: {collector.skipped}")
    
    # Run profiling
    profiling_data = run_performance_profiling(device)
    
    # Write report if requested
    if args.report:
        report_data = {
            "summary": {
                "total_tests": collector.total,
                "passed": collector.passed,
                "failed": collector.failed,
                "skipped": collector.skipped,
                "exit_code": int(exit_code)
            },
            "profiling": profiling_data
        }
        with open(args.report, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"\nPerformance and test report written to: {args.report}")
        
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
