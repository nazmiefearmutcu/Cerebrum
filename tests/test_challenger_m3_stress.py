import os
import sys
import resource
import pytest
import numpy as np
from datetime import datetime

# Adjust path to import from Cerebrum directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from power_parser import (
    KalmanFilter,
    MovingAverageFilter,
    parse_power_watts,
    parse_tegrastats_line,
    calculate_energy,
    lazy_tegrastats_reader
)

def test_memory_overhead_o1(monkeypatch):
    """
    Verify that memory overhead is truly O(1) by running a generator
    of 2,000,000 lines and measuring process RSS growth.
    """
    # 2,000,000 readings of 4.2W power
    def mock_reader(filepath):
        for i in range(2000000):
            yield (float(i), 4.2)
            
    monkeypatch.setattr("power_parser.lazy_tegrastats_reader", mock_reader)
    
    # Measure memory before
    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    # Run the energy calculation
    energy, mean_p, peak_p = calculate_energy("dummy_path", filter_type="none")
    
    # Measure memory after
    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    # On macOS, ru_maxrss is in bytes. On Linux, it is in kilobytes.
    # We check if memory growth is less than 5MB (which is very safe for O(1) processing of 2M lines).
    growth_bytes = mem_after - mem_before
    # If growth is negative (GC cleaned up), growth is 0.
    growth_mb = max(0.0, growth_bytes / (1024 * 1024))
    
    print(f"Memory before: {mem_before}, memory after: {mem_after}, growth: {growth_mb:.4f} MB")
    assert growth_mb < 5.0, f"Memory growth is too high: {growth_mb:.2f} MB"
    assert mean_p == pytest.approx(4.2, abs=1e-5)
    assert peak_p == pytest.approx(4.2, abs=1e-5)


def test_trapezoidal_irregular_sampling_validity(tmp_path):
    """
    Verify the physical validity of power integration outputs under irregular sampling.
    We define P(t) = 3.0 * t + 2.0 (linear power draw).
    Integral of P(t) from t=0 to t=10 is 0.5 * (P(0) + P(10)) * 10 = 0.5 * (2.0 + 32.0) * 10 = 170.0 J.
    Since P(t) is linear, trapezoidal integration should be EXACT.
    We will write irregular timestamps: t = [0.0, 1.5, 2.2, 5.0, 8.1, 10.0].
    """
    t_vals = [0.0, 1.5, 2.2, 5.0, 8.1, 10.0]
    p_vals = [3.0 * t + 2.0 for t in t_vals]
    
    log_file = tmp_path / "irregular_linear.log"
    with open(log_file, "w") as f:
        for t, p in zip(t_vals, p_vals):
            # p is in Watts, we write in mW (multiply by 1000)
            f.write(f"[{t}] VDD_IN {round(p * 1000)}mW\n")
            
    # Calculate energy
    energy, mean_p, peak_p = calculate_energy(str(log_file), filter_type="none", max_dt_ceiling=20.0)
    
    # Exact trapezoidal integral calculation
    expected_energy = 0.0
    for i in range(len(t_vals) - 1):
        dt = t_vals[i+1] - t_vals[i]
        expected_energy += 0.5 * (p_vals[i] + p_vals[i+1]) * dt
        
    assert expected_energy == pytest.approx(170.0, abs=1e-5)
    assert energy == pytest.approx(expected_energy, abs=1e-5)
    assert peak_p == pytest.approx(32.0, abs=1e-5)


def test_extreme_reboot_and_discontinuity_handling(tmp_path):
    """
    Verify behavior under extremely large reboots or clock resets.
    Scenario 1: Large gap (> max_dt_ceiling).
    Scenario 2: Negative gap (reboot / clock reset backward).
    Scenario 3: Extremely large timestamp values (e.g. 1e12 seconds) followed by normal timestamps.
    """
    # Scenario 1 & 2: Gap resetting
    log_file = tmp_path / "reboots.log"
    with open(log_file, "w") as f:
        f.write("[100.0] VDD_IN 4000mW\n")
        f.write("[102.0] VDD_IN 4000mW\n")   # dt = 2.0 -> 8.0J
        f.write("[1000.0] VDD_IN 4000mW\n")  # dt = 898.0 > max_dt_ceiling=5.0 -> Reset! Segment energy 0
        f.write("[1001.0] VDD_IN 4000mW\n")  # dt = 1.0 -> 4.0J
        f.write("[50.0] VDD_IN 4000mW\n")    # dt = -951.0 < 0 -> Reset! Segment energy 0
        f.write("[51.0] VDD_IN 4000mW\n")    # dt = 1.0 -> 4.0J
        
    # Expected energy: 8.0J + 4.0J + 4.0J = 16.0J
    energy, _, _ = calculate_energy(str(log_file), filter_type="none", max_dt_ceiling=5.0)
    assert energy == pytest.approx(16.0, abs=1e-5)

    # Scenario 3: Extremely large timestamp values
    log_file_large = tmp_path / "large_timestamps.log"
    with open(log_file_large, "w") as f:
        f.write("[1000000000000.0] VDD_IN 5000mW\n")
        f.write("[1000000000002.0] VDD_IN 5000mW\n") # dt = 2.0 -> 10.0J
        f.write("[1000000000000.0] VDD_IN 5000mW\n") # dt = -2.0 < 0 -> Reset!
        f.write("[1000000000001.0] VDD_IN 5000mW\n") # dt = 1.0 -> 5.0J
        
    energy_large, _, _ = calculate_energy(str(log_file_large), filter_type="none", default_dt=1.0)
    assert energy_large == pytest.approx(15.0, abs=1e-5)


def test_high_frequency_floating_point_absorption(tmp_path):
    """
    Test integrator behavior under extremely high frequency logs.
    100,000 logs at dt = 1e-4 seconds.
    Total duration = 10s. Power = 4.0W.
    Expected energy = 40.0J.
    Under float32 this might lose precision, but python floats are float64, so it should be fine.
    Let's verify.
    """
    log_file = tmp_path / "high_freq.log"
    with open(log_file, "w") as f:
        for idx in range(100000):
            t = idx * 0.0001
            f.write(f"[{t:.4f}] VDD_IN 4000mW\n")
            
    energy, mean_p, _ = calculate_energy(str(log_file), filter_type="none")
    assert energy == pytest.approx(40.0, abs=1e-3)
    assert mean_p == pytest.approx(4.0, abs=1e-5)


def test_noise_spike_filtering_behavior(tmp_path):
    """
    Test how Kalman and Moving Average filters handle a noise spike that is
    below the hard max_power_limit but represents significant noise.
    Baseline power = 4.0W. Spike = 30.0W (max_power_limit = 50.0).
    """
    log_file = tmp_path / "spikes.log"
    # Write: 4.0W, 4.0W, 4.0W, 30.0W (spike), 4.0W, 4.0W, 4.0W
    with open(log_file, "w") as f:
        f.write("[0.0] VDD_IN 4000mW\n")
        f.write("[1.0] VDD_IN 4000mW\n")
        f.write("[2.0] VDD_IN 4000mW\n")
        f.write("[3.0] VDD_IN 30000mW\n")
        f.write("[4.0] VDD_IN 4000mW\n")
        f.write("[5.0] VDD_IN 4000mW\n")
        f.write("[6.0] VDD_IN 4000mW\n")

    # Without filter: peak_p is 30.0
    _, _, peak_none = calculate_energy(str(log_file), filter_type="none")
    assert peak_none == 30.0

    # With Moving Average (window=3): spike is smoothed to (4.0+4.0+30.0)/3 = 12.67W
    _, _, peak_ma = calculate_energy(str(log_file), filter_type="moving_average", window_size=3)
    assert peak_ma == pytest.approx(12.666666, abs=1e-3)

    # With Kalman filter (q=1e-4, r=1e-2): spike should be heavily damped
    _, _, peak_kf = calculate_energy(str(log_file), filter_type="kalman", process_variance=1e-4, measurement_variance=1e-2)
    assert peak_kf < 15.0


def test_anomalous_zero_and_negative_inputs_corrected(tmp_path):
    """
    Verify how the parser handles zero and negative inputs.
    Zero inputs:
      - VDD_IN 0mW -> parsed to 0.0 -> detected as <= 0.0 -> imputed to prev_power (or 4.2). Correct.
    Negative inputs:
      - VDD_IN -3000mW -> parsed to -3.0.
      - Therefore, it is detected as an anomaly (<= 0.0) and is imputed.
    """
    # 1. Verify parsing behavior directly
    assert parse_power_watts("VDD_IN 0mW") == 0.0
    assert parse_power_watts("VDD_IN -3000mW") == -3.0

    # 2. Verify integration anomaly handling for zero
    log_file_zero = tmp_path / "zero.log"
    with open(log_file_zero, "w") as f:
        f.write("[0.0] VDD_IN 4000mW\n")
        f.write("[1.0] VDD_IN 0mW\n")  # zero, should be imputed to prev_power (4.0)
    
    energy_z, mean_z, peak_z = calculate_energy(str(log_file_zero), filter_type="none", default_dt=1.0)
    # Both readings are treated as 4.0. Energy = 0.5 * (4.0 + 4.0) * 1.0 = 4.0J.
    assert energy_z == pytest.approx(4.0, abs=1e-5)

    # 3. Verify integration anomaly handling for negative
    log_file_neg = tmp_path / "neg.log"
    with open(log_file_neg, "w") as f:
        f.write("[0.0] VDD_IN 4000mW\n")
        f.write("[1.0] VDD_IN -3000mW\n") # parsed as -3.0W (<= 0.0), so imputed to 4.0W!
        
    energy_n, mean_n, peak_n = calculate_energy(str(log_file_neg), filter_type="none", default_dt=1.0)
    # Both readings are treated as 4.0. Energy = 0.5 * (4.0 + 4.0) * 1.0 = 4.0J.
    assert energy_n == pytest.approx(4.0, abs=1e-5)
    assert mean_n == pytest.approx(4.0, abs=1e-5)

