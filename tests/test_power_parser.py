import os
import math
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
    assert parse_power_watts("VDD_IN 4.5W") == 4.5
    assert parse_power_watts("VDD_IN -3000mW") == -3.0
    assert parse_power_watts("POM_5V_IN 5.5W") == 5.5
    assert parse_power_watts("POM_5V_IN -4.5W") == -4.5
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
    # 0mW (imputed to 4.0), -3W (imputed to 4.0).
    # Time 0: 4W, Time 1: 4.0W, Time 2: 4.0W
    # dt=1: segment 1 = 0.5 * (4.0 + 4.0) * 1 = 4.0J
    # dt=1: segment 2 = 0.5 * (4.0 + 4.0) * 1 = 4.0J
    # Total energy = 8.0J
    assert energy == pytest.approx(8.0, abs=1e-5)
    assert mean_p == pytest.approx(4.0, abs=1e-5)
    assert peak_p == 4.0


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


def test_interface_contracts(tmp_path):
    from power_parser import parse_log_generator, compute_energy
    log_file = tmp_path / "test_contracts.log"
    with open(log_file, "w") as f:
        f.write("[0.0] VDD_IN 4200mW\n")
        f.write("[1.0] VDD_IN 4200mW\n")
        
    generator = parse_log_generator(str(log_file))
    assert hasattr(generator, "__next__")
    results = list(generator)
    assert len(results) == 2
    assert results[0][1] == 4.2
    
    energy = compute_energy(str(log_file))
    assert energy == pytest.approx(4.2)


def test_calculate_energy_memory_efficiency(monkeypatch):
    # Mock lazy_tegrastats_reader to yield 1,000,000 values
    def mock_reader(filepath):
        for i in range(1000000):
            yield (float(i), 4.2)
            
    monkeypatch.setattr("power_parser.lazy_tegrastats_reader", mock_reader)
    
    # Running O(1) calculate_energy under massive dataset should succeed quickly
    energy, mean_p, peak_p = calculate_energy("dummy_path", filter_type="none")
    assert mean_p == pytest.approx(4.2, abs=1e-5)
    assert peak_p == pytest.approx(4.2, abs=1e-5)
    
    # Verify with Kalman Filter too
    energy_kf, mean_kf, peak_kf = calculate_energy("dummy_path", filter_type="kalman")
    assert mean_kf == pytest.approx(4.2, abs=0.1)


def test_discontinuity_handling(tmp_path):
    log_file = tmp_path / "tegrastats_discontinuity.log"
    # Write entries with gaps:
    # t=0: 4.0W
    # t=1: 4.0W (dt=1.0) -> 4.0J
    # t=10: 4.0W (dt=9.0 > max_dt_ceiling=5.0) -> Discontinuity!
    #   Should reset integration chain, adding 0J for this gap.
    # t=11: 4.0W (dt=1.0) -> 4.0J
    # Total energy should be 4.0J + 4.0J = 8.0J (instead of 44.0J if integrated continuously)
    with open(log_file, "w") as f:
        f.write("[0.0] VDD_IN 4000mW\n")
        f.write("[1.0] VDD_IN 4000mW\n")
        f.write("[10.0] VDD_IN 4000mW\n")
        f.write("[11.0] VDD_IN 4000mW\n")
        
    energy, mean_p, peak_p = calculate_energy(str(log_file), filter_type="none", max_dt_ceiling=5.0)
    assert energy == pytest.approx(8.0, abs=1e-5)

    # Test negative dt discontinuity
    # t=0: 4.0W
    # t=1: 4.0W (dt=1) -> 4J
    # t=0.5: 4.0W (dt=-0.5 < 0) -> Discontinuity! Reset.
    # t=1.5: 4.0W (dt=1.0) -> 4J
    # Total energy should be 4.0J + 4.0J = 8.0J
    log_file_neg = tmp_path / "tegrastats_discontinuity_neg.log"
    with open(log_file_neg, "w") as f:
        f.write("[0.0] VDD_IN 4000mW\n")
        f.write("[1.0] VDD_IN 4000mW\n")
        f.write("[0.5] VDD_IN 4000mW\n")
        f.write("[1.5] VDD_IN 4000mW\n")
        
    energy_neg, _, _ = calculate_energy(str(log_file_neg), filter_type="none", default_dt=1.0)
    assert energy_neg == pytest.approx(8.0, abs=1e-5)


def test_kalman_time_varying():
    # If dt is larger, process noise covariance q * dt increases, so the filter trusts measurement more
    kf_small = KalmanFilter(q=1e-2, r=1e-2, initial_value=4.0)
    kf_large = KalmanFilter(q=1e-2, r=1e-2, initial_value=4.0)
    kf_small.p = 1e-4
    kf_large.p = 1e-4
    
    val_small = kf_small.filter(10.0, dt=1.0)
    val_large = kf_large.filter(10.0, dt=10.0)
    
    assert val_large > val_small
    assert val_large <= 10.0
    assert val_small > 4.0


def test_baseline_command():
    # Verify we can execute with python power_parser.py --baseline
    cmd = [sys.executable, "power_parser.py", "--baseline"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "STATUS: PASS" in result.stdout


def test_filter_guards():
    # Kalman filter division-by-zero guard test
    kf = KalmanFilter(q=0.0, r=0.0, initial_value=4.0)
    kf.p = 0.0
    # Should not crash and should return x
    res = kf.filter(5.0, dt=1.0)
    assert res == 4.0

    # MovingAverageFilter window size guard test
    ma = MovingAverageFilter(window_size=0)
    assert ma.window_size == 1
    assert ma.filter(5.0) == 5.0


def test_scientific_notation_parsing():
    assert parse_power_watts("VDD_IN 1e20mW") == pytest.approx(1e17)
    assert parse_power_watts("VDD_IN 1.2e4mW") == pytest.approx(12.0)
    assert parse_power_watts("VDD_IN 2.0e1W") == pytest.approx(20.0)
    assert parse_power_watts("VDD_IN 1E2mw") == pytest.approx(0.1)
    assert parse_power_watts("VDD_IN 3.5e-1w") == pytest.approx(0.35)


def test_nan_inf_parsing_and_imputation(tmp_path):
    assert np.isnan(parse_power_watts("VDD_IN NaNmW"))
    assert parse_power_watts("VDD_IN InfW") == float('inf')
    assert parse_power_watts("VDD_IN -inFw") == float('-inf')
    
    log_file = tmp_path / "tegrastats_nan_inf.log"
    with open(log_file, "w") as f:
        f.write("[0.0] VDD_IN 4200mW\n")  # 4.2W
        f.write("[1.0] VDD_IN NaNmW\n")   # imputed to 4.2W
        f.write("[2.0] VDD_IN InfW\n")    # imputed to 4.2W
        f.write("[3.0] VDD_IN -InfW\n")   # imputed to 4.2W
        
    energy, mean_p, peak_p = calculate_energy(str(log_file), filter_type="none", default_dt=1.0)
    assert energy == pytest.approx(12.6, abs=1e-5)
    assert mean_p == pytest.approx(4.2, abs=1e-5)
    assert peak_p == 4.2


def test_filter_transient_step_responses():
    # 1. Moving Average Filter step change from 4.2 to 25.0
    ma = MovingAverageFilter(window_size=5)
    for _ in range(5):
        ma.filter(4.2)
    assert ma.filter(4.2) == pytest.approx(4.2)
    
    outputs_ma = []
    for _ in range(5):
        outputs_ma.append(ma.filter(25.0))
        
    assert outputs_ma[0] > 4.2
    assert outputs_ma[-1] == pytest.approx(25.0)
    
    # 2. Kalman Filter step change from 4.2 to 25.0
    kf = KalmanFilter(q=1e-2, r=1e-2, initial_value=4.2)
    outputs_kf = []
    for _ in range(20):
        outputs_kf.append(kf.filter(25.0, dt=1.0))
        
    for i in range(1, len(outputs_kf)):
        assert outputs_kf[i] > outputs_kf[i-1]
    assert outputs_kf[-1] == pytest.approx(25.0, abs=0.5)


def test_malformed_numeric_formats_stress(tmp_path):
    import math
    from power_parser import parse_power_watts, calculate_energy

    # Test direct parsing of various forms
    assert parse_power_watts("VDD_IN 1.2e-10W") == pytest.approx(1.2e-10)
    assert parse_power_watts("VDD_IN 3.5e-300mW") == pytest.approx(3.5e-303)
    assert parse_power_watts("VDD_IN 1e300mW") == pytest.approx(1e297)
    assert parse_power_watts("VDD_IN    4200   mW") == pytest.approx(4.2)
    assert parse_power_watts("VDD_IN\t9.5\tW") == pytest.approx(9.5)
    assert parse_power_watts("VDD_IN 12.0  ") == pytest.approx(0.012)  # default mW when no unit is VDD_IN 12.0 -> returns val/1000.0

    # Ensure NaN/Inf variants parsed correctly (case insensitive)
    assert np.isnan(parse_power_watts("VDD_IN NaNmW"))
    assert np.isnan(parse_power_watts("VDD_IN nan"))
    assert parse_power_watts("VDD_IN -infinityW") == float('-inf')
    assert parse_power_watts("VDD_IN +InF") == float('inf')

    # Now write a file with these values mixed in and calculate energy.
    # We want to ensure that it computes energy safely without crashing,
    # and does not pollute the integrated energy (i.e. the extreme values/NaNs/Infs are imputed).
    log_file = tmp_path / "tegrastats_stress.log"
    with open(log_file, "w") as f:
        # Time steps of 1.0s
        f.write("[0.0] VDD_IN 4200mW\n")            # 4.2W
        f.write("[1.0] VDD_IN 1e300mW\n")           # 1e297W -> anomalous (too large) -> imputed to prev (4.2W)
        f.write("[2.0] VDD_IN 1.2e-10W\n")          # 1.2e-10W -> valid (very small)
        f.write("[3.0] VDD_IN    4200   mW\n")      # 4.2W -> valid (spacing)
        f.write("[4.0] VDD_IN\t9.5\tW\n")           # 9.5W -> valid (tabs)
        f.write("[5.0] VDD_IN NaNmW\n")             # NaN -> anomalous -> imputed to prev (9.5W)
        f.write("[6.0] VDD_IN -infinityW\n")        # -inf -> anomalous -> imputed to prev (9.5W)
        f.write("[7.0] VDD_IN +InF\n")              # +inf -> anomalous -> imputed to prev (9.5W)
        f.write("[8.0] VDD_IN -5.0W\n")             # negative -> anomalous -> imputed to prev (9.5W)
        
    energy, mean_p, peak_p = calculate_energy(str(log_file), filter_type="none", default_dt=1.0, max_power_limit=50.0)
    
    # Let's trace the power values after imputation:
    # t=0: 4.2
    # t=1: 1e297 (anomalous) -> imputed to 4.2. Current power p = 4.2
    #   dt = 1.0. Energy step: 0.5 * (4.2 + 4.2) * 1.0 = 4.2 J. Total energy = 4.2 J.
    # t=2: 1.2e-10. Valid. Current power p = 1.2e-10
    #   dt = 1.0. Energy step: 0.5 * (4.2 + 1.2e-10) * 1.0 = 2.1 J. Total energy = 6.3 J.
    # t=3: 4.2. Valid. Current power p = 4.2
    #   dt = 1.0. Energy step: 0.5 * (1.2e-10 + 4.2) * 1.0 = 2.1 J. Total energy = 8.4 J.
    # t=4: 9.5. Valid. Current power p = 9.5
    #   dt = 1.0. Energy step: 0.5 * (4.2 + 9.5) * 1.0 = 6.85 J. Total energy = 15.25 J.
    # t=5: NaN (anomalous) -> imputed to 9.5. Current power p = 9.5
    #   dt = 1.0. Energy step: 0.5 * (9.5 + 9.5) * 1.0 = 9.5 J. Total energy = 24.75 J.
    # t=6: -inf (anomalous) -> imputed to 9.5. Current power p = 9.5
    #   dt = 1.0. Energy step: 0.5 * (9.5 + 9.5) * 1.0 = 9.5 J. Total energy = 34.25 J.
    # t=7: +inf (anomalous) -> imputed to 9.5. Current power p = 9.5
    #   dt = 1.0. Energy step: 0.5 * (9.5 + 9.5) * 1.0 = 9.5 J. Total energy = 43.75 J.
    # t=8: -5.0 (anomalous) -> imputed to 9.5. Current power p = 9.5
    #   dt = 1.0. Energy step: 0.5 * (9.5 + 9.5) * 1.0 = 9.5 J. Total energy = 53.25 J.
    #
    assert math.isfinite(energy)
    assert energy == pytest.approx(53.25, abs=1e-5)
    assert mean_p == pytest.approx((4.2 + 4.2 + 1.2e-10 + 4.2 + 9.5 + 9.5 + 9.5 + 9.5 + 9.5) / 9, abs=1e-5)
    assert peak_p == 9.5



