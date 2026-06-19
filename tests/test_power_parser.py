import os
import pytest
import numpy as np
from datetime import datetime
import subprocess
import sys

from power_parser import (
    KalmanFilter,
    MovingAverageFilter,
    parse_power_watts,
    parse_tegrastats_line,
    calculate_energy,
    generate_baseline_log,
    lazy_tegrastats_reader
)

def test_parse_power_watts():
    # Test different line formats
    assert parse_power_watts("VDD_IN 4200mW") == 4.2
    assert parse_power_watts("POM_5V_IN 5.5W") == 5.5
    assert parse_power_watts("some noise 1500mW and more") == 1.5
    assert parse_power_watts("no power info") == 4.2  # default fallback


def test_parse_tegrastats_line():
    line = "[2026-06-19 16:30:15.123] VDD_IN 4200mW"
    ts, power = parse_tegrastats_line(line)
    assert ts is not None
    assert power == 4.2


def test_kalman_filter():
    kf = KalmanFilter(q=1e-4, r=1e-2, initial_value=4.0)
    # A sequence of measurements with a noise spike
    measurements = [4.0, 4.1, 4.0, 10.0, 4.0, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1]
    filtered = [kf.filter(m) for m in measurements]
    # The noise spike at index 3 (10.0) should be heavily dampened
    assert filtered[3] < 7.0
    assert filtered[-1] == pytest.approx(4.1, abs=0.5)



def test_moving_average_filter():
    ma = MovingAverageFilter(window_size=3)
    assert ma.filter(1.0) == 1.0
    assert ma.filter(2.0) == 1.5
    assert ma.filter(3.0) == 2.0
    assert ma.filter(4.0) == 3.0  # window has [2, 3, 4]


def test_energy_calculation_trapezoidal(tmp_path):
    log_file = tmp_path / "tegrastats.log"
    # Create a test log with irregular sampling
    base_time = datetime(2026, 6, 19, 16, 30, 0).timestamp()
    
    # Write entries with irregular intervals
    with open(log_file, "w") as f:
        # Time 0: 4W
        f.write(f"[{base_time}] VDD_IN 4000mW\n")
        # Time 2: 6W (dt = 2.0s) -> segment energy = 0.5 * (4 + 6) * 2 = 10J
        f.write(f"[{base_time + 2.0}] VDD_IN 6000mW\n")
        # Time 3: 4W (dt = 1.0s) -> segment energy = 0.5 * (6 + 4) * 1 = 5J
        # Total energy = 15J
        f.write(f"[{base_time + 3.0}] VDD_IN 4000mW\n")
        
    energy, mean_p, peak_p = calculate_energy(str(log_file), filter_type="none")
    assert energy == pytest.approx(15.0, abs=1e-5)
    assert mean_p == pytest.approx(14/3, abs=1e-5)
    assert peak_p == 6.0


def test_zero_and_negative_power_handling(tmp_path):
    log_file = tmp_path / "tegrastats_anomalies.log"
    with open(log_file, "w") as f:
        f.write("[0.0] VDD_IN 4000mW\n")
        f.write("[1.0] VDD_IN 0mW\n")      # zero power anomaly
        f.write("[2.0] VDD_IN -3000mW\n")   # negative power anomaly
        
    energy, mean_p, peak_p = calculate_energy(str(log_file), filter_type="none", default_dt=1.0)
    # 0mW becomes 0.1W, -3W becomes 3W.
    # Time 0: 4W, Time 1: 0.1W, Time 2: 3.0W
    # dt=1: segment 1 = 0.5 * (4 + 0.1) * 1 = 2.05J
    # dt=1: segment 2 = 0.5 * (0.1 + 3) * 1 = 1.55J
    # Total energy = 3.6J
    assert energy == pytest.approx(3.6, abs=1e-5)


def test_lazy_reader_memory_footprint(tmp_path):
    log_file = tmp_path / "large_tegrastats.log"
    # Write 1000 lines
    with open(log_file, "w") as f:
        for idx in range(1000):
            f.write(f"[{idx}] VDD_IN 4200mW\n")
            
    generator = lazy_tegrastats_reader(str(log_file))
    # Assert it returns a generator
    assert hasattr(generator, "__next__")
    
    # Check we can consume
    lines_parsed = 0
    for ts, p in generator:
        lines_parsed += 1
        assert p == 4.2
    assert lines_parsed == 1000


def test_baseline_command():
    # Verify we can execute with python power_parser.py --baseline
    cmd = [sys.executable, "power_parser.py", "--baseline"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "STATUS: PASS" in result.stdout
