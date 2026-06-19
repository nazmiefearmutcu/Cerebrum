#!/usr/bin/env python3
"""
Custom Test Runner and Stress Suite for Tegrastats Power Parser.
Runs under python3 -S (sandboxed, zero external dependencies).
Uses local files in tmp/ instead of tempfile to comply with sandbox constraints.
Catches all exceptions to avoid unhandled tracebacks that trigger the sandbox.
"""

import os
import sys
import tracemalloc
from datetime import datetime

# Insert root directory into path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import power_parser
from power_parser import (
    KalmanFilter,
    MovingAverageFilter,
    parse_power_watts,
    parse_tegrastats_line,
    calculate_energy,
    generate_baseline_log,
    lazy_tegrastats_reader
)

def approx(val, target, abs_tol=1e-5):
    return abs(val - target) <= abs_tol

def ensure_tmp_dir():
    if not os.path.exists("tmp"):
        os.makedirs("tmp")

# ----------------- Unit Tests (Replicating pytest tests) -----------------

def test_parse_power_watts():
    assert approx(parse_power_watts("VDD_IN 4200mW"), 4.2)
    assert approx(parse_power_watts("POM_5V_IN 5.5W"), 5.5)
    assert approx(parse_power_watts("some noise 1500mW and more"), 1.5)
    assert approx(parse_power_watts("no power info"), 4.2)  # default fallback

def test_parse_tegrastats_line():
    line = "[2026-06-19 16:30:15.123] VDD_IN 4200mW"
    ts, power = parse_tegrastats_line(line)
    assert ts is not None
    assert approx(power, 4.2)

def test_kalman_filter():
    kf = KalmanFilter(q=1e-4, r=1e-2, initial_value=4.0)
    measurements = [4.0, 4.1, 4.0, 10.0, 4.0, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1]
    filtered = [kf.filter(m) for m in measurements]
    assert filtered[3] < 7.0
    assert approx(filtered[-1], 4.1, abs_tol=0.5)

def test_moving_average_filter():
    ma = MovingAverageFilter(window_size=3)
    assert approx(ma.filter(1.0), 1.0)
    assert approx(ma.filter(2.0), 1.5)
    assert approx(ma.filter(3.0), 2.0)
    assert approx(ma.filter(4.0), 3.0)

def test_energy_calculation_trapezoidal():
    ensure_tmp_dir()
    log_file = "tmp/test_energy_calc.log"
    base_time = datetime(2026, 6, 19, 16, 30, 0).timestamp()
    with open(log_file, "w") as f:
        f.write(f"[{base_time}] VDD_IN 4000mW\n")
        f.write(f"[{base_time + 2.0}] VDD_IN 6000mW\n")
        f.write(f"[{base_time + 3.0}] VDD_IN 4000mW\n")
        
    try:
        energy, mean_p, peak_p = calculate_energy(log_file, filter_type="none")
        assert approx(energy, 15.0)
        assert approx(mean_p, 14.0/3.0)
        assert approx(peak_p, 6.0)
    finally:
        if os.path.exists(log_file):
            os.remove(log_file)

def test_zero_and_negative_power_handling():
    ensure_tmp_dir()
    log_file = "tmp/test_zero_neg.log"
    with open(log_file, "w") as f:
        f.write("[0.0] VDD_IN 4000mW\n")
        f.write("[1.0] VDD_IN 0mW\n")
        f.write("[2.0] VDD_IN -3000mW\n")
        
    try:
        energy, mean_p, peak_p = calculate_energy(log_file, filter_type="none", default_dt=1.0)
        # The worker's test asserts 8.0J (assuming -3000mW is imputed to 4.0W).
        # We assert 8.0J to check if the implementation matches this correct physical behavior.
        # (It will fail because of the negative parsing bug which yields 7.5J)
        assert approx(energy, 8.0), f"Expected 8.0J, got {energy}J"
        assert approx(mean_p, 4.0), f"Expected 4.0W, got {mean_p}W"
        assert approx(peak_p, 4.0), f"Expected 4.0W, got {peak_p}W"
    finally:
        if os.path.exists(log_file):
            os.remove(log_file)

def test_lazy_reader_memory_footprint():
    ensure_tmp_dir()
    log_file = "tmp/test_lazy_footprint.log"
    with open(log_file, "w") as f:
        for idx in range(1000):
            f.write(f"[{idx}] VDD_IN 4200mW\n")
            
    try:
        generator = lazy_tegrastats_reader(log_file)
        assert hasattr(generator, "__next__")
        lines_parsed = 0
        for ts, p in generator:
            lines_parsed += 1
            assert approx(p, 4.2)
        assert lines_parsed == 1000
    finally:
        if os.path.exists(log_file):
            os.remove(log_file)

def test_interface_contracts():
    ensure_tmp_dir()
    log_file = "tmp/test_intf_contracts.log"
    with open(log_file, "w") as f:
        f.write("[0.0] VDD_IN 4200mW\n")
        f.write("[1.0] VDD_IN 4200mW\n")
        
    try:
        from power_parser import parse_log_generator, compute_energy
        generator = parse_log_generator(log_file)
        assert hasattr(generator, "__next__")
        results = list(generator)
        assert len(results) == 2
        assert approx(results[0][1], 4.2)
        
        energy = compute_energy(log_file)
        assert approx(energy, 4.2)
    finally:
        if os.path.exists(log_file):
            os.remove(log_file)

def test_discontinuity_handling():
    ensure_tmp_dir()
    log_file = "tmp/test_discont.log"
    with open(log_file, "w") as f:
        f.write("[0.0] VDD_IN 4000mW\n")
        f.write("[1.0] VDD_IN 4000mW\n")
        f.write("[10.0] VDD_IN 4000mW\n")
        f.write("[11.0] VDD_IN 4000mW\n")
        
    try:
        energy, mean_p, peak_p = calculate_energy(log_file, filter_type="none", max_dt_ceiling=5.0)
        assert approx(energy, 8.0)
    finally:
        if os.path.exists(log_file):
            os.remove(log_file)

    log_file_neg = "tmp/test_discont_neg.log"
    with open(log_file_neg, "w") as f:
        f.write("[0.0] VDD_IN 4000mW\n")
        f.write("[1.0] VDD_IN 4000mW\n")
        f.write("[0.5] VDD_IN 4000mW\n")
        f.write("[1.5] VDD_IN 4000mW\n")
        
    try:
        energy_neg, _, _ = calculate_energy(log_file_neg, filter_type="none", default_dt=1.0)
        assert approx(energy_neg, 8.0)
    finally:
        if os.path.exists(log_file_neg):
            os.remove(log_file_neg)

def test_kalman_time_varying():
    kf_small = KalmanFilter(q=1e-2, r=1e-2, initial_value=4.0)
    kf_large = KalmanFilter(q=1e-2, r=1e-2, initial_value=4.0)
    kf_small.p = 1e-4
    kf_large.p = 1e-4
    
    val_small = kf_small.filter(10.0, dt=1.0)
    val_large = kf_large.filter(10.0, dt=10.0)
    
    assert val_large > val_small
    assert val_large <= 10.0
    assert val_small > 4.0


# ----------------- Empirical Challenger Stress & Adversarial Tests -----------------

def test_memory_overhead_stress():
    # Mock a generator that yields 500,000 power lines
    def mock_large_generator(filepath):
        for idx in range(500000):
            yield (float(idx), 4.2)
            
    original_reader = power_parser.lazy_tegrastats_reader
    power_parser.lazy_tegrastats_reader = mock_large_generator
    
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    
    # Run integration over 500k data points
    energy, mean_p, peak_p = calculate_energy("dummy", filter_type="none")
    
    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    
    power_parser.lazy_tegrastats_reader = original_reader
    
    # Calculate memory allocated during the process
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_diff = sum(stat.size_diff for stat in stats)
    
    assert total_diff < 100 * 1024, f"Memory overhead too high: {total_diff} bytes"

def test_kalman_division_by_zero():
    kf = KalmanFilter(q=1e-4, r=0.0, initial_value=4.0)
    kf.p = 0.0  # Force covariance to 0
    # This must raise ZeroDivisionError due to vulnerability
    try:
        kf.filter(5.0, dt=0.0)
        raise AssertionError("KalmanFilter did not raise ZeroDivisionError for r=0, dt=0")
    except ZeroDivisionError:
        pass

def test_moving_average_division_by_zero():
    ma = MovingAverageFilter(window_size=0)
    try:
        ma.filter(1.0)
        raise AssertionError("MovingAverageFilter did not raise ZeroDivisionError for window_size=0")
    except ZeroDivisionError:
        pass

def test_negative_window_size():
    ma = MovingAverageFilter(window_size=-2)
    try:
        ma.filter(1.0)
        ma.filter(2.0)
        ma.filter(3.0)
        raise AssertionError("MovingAverageFilter did not raise ZeroDivisionError for window_size < 0")
    except ZeroDivisionError:
        pass

def test_decimal_watt_parsing_bug():
    val = parse_power_watts("VDD_IN 4.5W")
    # This must fail to match correct behavior (returns 0.004W instead of 4.5W)
    assert approx(val, 4.5), f"Expected 4.5W, got {val}W"

def test_irregular_sampling_stress():
    ensure_tmp_dir()
    log_file = "tmp/test_irregular_stress.log"
    with open(log_file, "w") as f:
        f.write("[0.0] VDD_IN 4000mW\n")
        f.write("[0.0001] VDD_IN 4100mW\n")
        f.write("[0.0002] VDD_IN 4200mW\n")
        f.write("[4.9] VDD_IN 4000mW\n")
        f.write("[10.0] VDD_IN 4000mW\n")
        f.write("[11.0] VDD_IN 4000mW\n")
        
    try:
        energy, mean_p, peak_p = calculate_energy(log_file, filter_type="none", max_dt_ceiling=5.0)
        # Expected integration = 24.09J
        assert approx(energy, 24.09), f"Expected 24.09J, got {energy}J"
    finally:
        if os.path.exists(log_file):
            os.remove(log_file)

def test_extreme_noise_spikes():
    ensure_tmp_dir()
    log_file = "tmp/test_noise_spikes.log"
    with open(log_file, "w") as f:
        f.write("[0.0] VDD_IN 4000mW\n")
        f.write("[1.0] VDD_IN 1000000mW\n")
        f.write("[2.0] VDD_IN 4000mW\n")
        
    try:
        energy, mean_p, peak_p = calculate_energy(log_file, filter_type="none", max_power_limit=50.0)
        assert approx(energy, 8.0)
        assert approx(peak_p, 4.0)
    finally:
        if os.path.exists(log_file):
            os.remove(log_file)

def run_all():
    tests = [
        ("test_parse_power_watts", test_parse_power_watts),
        ("test_parse_tegrastats_line", test_parse_tegrastats_line),
        ("test_kalman_filter", test_kalman_filter),
        ("test_moving_average_filter", test_moving_average_filter),
        ("test_energy_calculation_trapezoidal", test_energy_calculation_trapezoidal),
        ("test_zero_and_negative_power_handling", test_zero_and_negative_power_handling),
        ("test_lazy_reader_memory_footprint", test_lazy_reader_memory_footprint),
        ("test_interface_contracts", test_interface_contracts),
        ("test_discontinuity_handling", test_discontinuity_handling),
        ("test_kalman_time_varying", test_kalman_time_varying),
        ("test_memory_overhead_stress", test_memory_overhead_stress),
        ("test_kalman_division_by_zero", test_kalman_division_by_zero),
        ("test_moving_average_division_by_zero", test_moving_average_division_by_zero),
        ("test_negative_window_size", test_negative_window_size),
        ("test_decimal_watt_parsing_bug", test_decimal_watt_parsing_bug),
        ("test_irregular_sampling_stress", test_irregular_sampling_stress),
        ("test_extreme_noise_spikes", test_extreme_noise_spikes)
    ]
    
    passed = 0
    failed = 0
    
    print("Starting Sandboxed Test Execution...\n")
    for name, test in tests:
        print(f"Running {name}...")
        try:
            test()
            passed += 1
            print("  --> PASS")
        except AssertionError as e:
            failed += 1
            print(f"  --> FAIL (AssertionError): {e}")
        except Exception as e:
            failed += 1
            print(f"  --> FAIL (Unexpected Exception): {type(e).__name__}: {e}")
            
    print(f"\nTest Summary: {passed} passed, {failed} failed.")
    if failed > 0:
        print("\nVerdict: FAIL")
    else:
        print("\nVerdict: PASS")

if __name__ == "__main__":
    run_all()
