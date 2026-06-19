"""
Pytest Suite for Sim2Real Physical Validation Components and Mocks.
"""

import pytest
import numpy as np
from cerebrum.config import CerebrumConfig
from cerebrum.unified import CerebrumNet

# Local relative imports in target runtime structure
from physical_validation import (
    TelemetryCalibrator, MotorMapper, GainScaler, InferenceTuner, WarmupRamper
)
from run_physical_validation import MockHardware, run_checklist

@pytest.fixture
def test_setup():
    cfg = CerebrumConfig(dims=(4, 6), n_settle=5, seed=123)
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg)
    calibrator = TelemetryCalibrator()
    mapper = MotorMapper()
    return net, calibrator, mapper


def test_telemetry_calibrator_nan_inf_removal(test_setup):
    _, calibrator, _ = test_setup
    raw_bad = {
        "lidar": np.array([np.nan, 2.0, np.inf, 4.0]),
        "camera": np.array([0.5, np.nan]),
        "odometry": np.array([np.inf, np.nan])
    }
    
    calibrated = calibrator.calibrate(raw_bad)
    
    # Assert NaN values are cleansed
    assert not np.isnan(calibrated["lidar"]).any()
    assert not np.isinf(calibrated["lidar"]).any()
    # Confirm maximum range replacement on bad lidar signals
    assert calibrated["lidar"][0] == 10.0
    assert calibrated["lidar"][2] == 10.0
    assert calibrated["camera"][1] == 0.0
    assert calibrated["odometry"][0] == 2.0  # saturated value limit


def test_motor_mapper_deadband_compensation():
    mapper = MotorMapper(deadband=0.1, max_wheel_vel=2.0)
    
    # 1. Output below deadband is zeroed
    small_speeds = mapper.workspace_to_wheel(0.002, 0.0)
    assert np.all(small_speeds == 0.0)
    
    # 2. Output above deadband passes through
    large_speeds = mapper.workspace_to_wheel(0.8, 0.0)
    assert np.any(large_speeds > 0.0)
    assert np.all(large_speeds <= 2.0)


def test_gain_scaler_tilt_damping():
    scaler = GainScaler(base_gain=1.0, tilt_hazard_threshold=0.5)
    
    # Flat posture -> full control gain
    gain_flat = scaler.compute_scaling_factor(current_tilt=0.0, reconstruction_error=0.0)
    assert gain_flat == 1.0
    
    # Steep posture tilt -> control gains are aggressively damped or zeroed out
    gain_tilt = scaler.compute_scaling_factor(current_tilt=0.45, reconstruction_error=0.0)
    assert gain_tilt < 0.5
    
    # Extreme tilt -> emergency lock
    gain_unsafe = scaler.compute_scaling_factor(current_tilt=0.6, reconstruction_error=0.0)
    assert gain_unsafe == scaler.min_gain


def test_inference_tuner_latency_control(test_setup):
    net, _, _ = test_setup
    tuner = InferenceTuner(target_hz=50.0, max_latency_sec=0.020, min_n_settle=3)
    
    # Force high latency loop reporting
    tuner.tune(net, last_step_duration=0.035)
    tuner.tune(net, last_step_duration=0.028)
    
    # n_settle should scale down to preserve control frequency constraints
    assert net.cfg.n_settle < 5
    assert net.cfg.n_settle >= 3


def test_warmup_ramper():
    ramper = WarmupRamper(ramp_steps=10)
    
    # Initial gain must start at 0.0
    assert ramper.get_warmup_multiplier() == 0.0
    
    # Iterate and confirm progressive ramp up
    for _ in range(5):
        ramper.step()
    assert ramper.get_warmup_multiplier() == 0.5
    
    for _ in range(10):
        ramper.step()
    assert ramper.get_warmup_multiplier() == 1.0
    assert ramper.is_complete()


def test_mock_hardware_dropouts():
    hw = MockHardware(dropout_probability=1.0)  # Complete packet loss
    hw.set_motor_commands(1.0, 1.5)
    
    # Commands should not update (ZOH holds zero initial state)
    assert hw.target_v_l == 0.0
    assert hw.target_v_r == 0.0
    
    # Telemetry should return None fields
    telem = hw.get_telemetry()
    assert all(val is None for val in telem.values())


def test_gain_scaler_surprise_damping():
    scaler = GainScaler(base_gain=1.0, surprise_hazard_threshold=5.0)
    
    # Surprise within bounds -> full control gain
    gain_normal = scaler.compute_scaling_factor(current_tilt=0.0, reconstruction_error=4.0)
    assert gain_normal == 1.0
    
    # Surprise out of bounds -> control gains are dynamically scaled down
    gain_surprise = scaler.compute_scaling_factor(current_tilt=0.0, reconstruction_error=10.0)
    assert gain_surprise == 0.5  # 5.0 / 10.0 = 0.5


def test_warmup_ramped_posture():
    ramper = WarmupRamper(ramp_steps=10, posture_start=-0.2, posture_target=0.8)
    
    # Initial posture is at start
    assert ramper.get_ramped_posture() == -0.2
    
    # Step half way
    for _ in range(5):
        ramper.step()
    assert pytest.approx(ramper.get_ramped_posture()) == 0.3
    
    # Step to completion
    for _ in range(5):
        ramper.step()
    assert pytest.approx(ramper.get_ramped_posture()) == 0.8


def test_motor_mapper_slew_rate():
    mapper = MotorMapper(max_slew_rate=10.0, max_wheel_vel=40.0)
    # Target 20 rad/s. At dt=0.02, max delta is 10 * 0.02 = 0.2 rad/s.
    cmd = mapper.workspace_to_wheel(1.0, 0.0, dt=0.02) # linear_v = 1.0 -> 20.0 rad/s target
    assert pytest.approx(cmd[0]) == 0.2
    assert pytest.approx(cmd[1]) == 0.2


def test_run_checklist(test_setup):
    net, calibrator, _ = test_setup
    hw = MockHardware(dropout_probability=0.0) # ensure no dropout during test
    checklist_res = run_checklist(hw, calibrator, net)
    assert checklist_res["status"] == "PASS"
    assert len(checklist_res["errors"]) == 0


def test_geometric_solver_singularity():
    from physical_validation import Geometric3DOFSolver
    solver = Geometric3DOFSolver(l1=1.0, l2=1.0, l3=1.0)
    
    # Target directly on Z-axis (x=0, y=0) should lock base angle to 0.0 without NaN
    angles, reachable = solver.inverse_kinematics(0.0, 0.0, 2.5)
    assert angles[0] == 0.0
    assert not np.isnan(angles).any()


def test_geometric_solver_out_of_bounds():
    from physical_validation import Geometric3DOFSolver
    solver = Geometric3DOFSolver(l1=1.0, l2=1.0, l3=1.0)
    
    # Target way out of reach (max reach is l2 + l3 = 2.0)
    angles, reachable = solver.inverse_kinematics(10.0, 10.0, 10.0)
    assert not reachable
    # Ensure no NaN is produced (safeguarded by arccos clipping)
    assert not np.isnan(angles).any()
    assert not np.isinf(angles).any()


def test_joint_to_ticks_conversion():
    from physical_validation import joint_to_ticks
    # Verify exact integer rounding
    assert isinstance(joint_to_ticks(1.2345), int)
    assert joint_to_ticks(1.2344, ticks_per_rad=1000.0) == 1234
    assert joint_to_ticks(1.2346, ticks_per_rad=1000.0) == 1235


def test_torque_to_current_clamping():
    from physical_validation import torque_to_current
    # Within bounds
    assert torque_to_current(2.0, torque_constant=0.5, max_current=10.0) == 4.0
    # Over bounds (should soft-clamp to max_current)
    assert torque_to_current(10.0, torque_constant=0.5, max_current=10.0) == 10.0
    assert torque_to_current(-10.0, torque_constant=0.5, max_current=10.0) == -10.0


def test_safe_get_telemetry_recovery():
    from physical_validation import safe_get_telemetry
    
    class GlitchyHardware:
        def __init__(self):
            self.calls = 0
        def get_telemetry(self):
            self.calls += 1
            if self.calls < 3:
                return None  # simulated temporary connection loss
            return {
                "lidar": np.array([2.0, 5.0, 5.0, 5.0]),
                "camera": np.array([0.8, 0.2]),
                "odometry": np.array([0.1, 0.0]),
                "tilt": np.array([0.0])
            }
            
    hw = GlitchyHardware()
    last_valid = {
        "lidar": np.array([1.5, 5.0, 5.0, 5.0]),
        "camera": np.array([0.7, 0.3]),
        "odometry": np.array([0.0, 0.0]),
        "tilt": np.array([0.0])
    }
    
    # 1. Recovery scenario: should retry, succeed, and return true status
    telem, ok = safe_get_telemetry(hw, last_valid_telemetry=last_valid, retries=5)
    assert ok
    assert telem["lidar"][0] == 2.0
    assert hw.calls == 3  # resolved on 3rd call
    
    # 2. Total failure scenario: exceeds retries, degrades gracefully to last valid
    hw_dead = GlitchyHardware() # always returns None
    telem_fallback, ok_fallback = safe_get_telemetry(hw_dead, last_valid_telemetry=last_valid, retries=2)
    assert not ok_fallback
    assert telem_fallback["lidar"][0] == 1.5  # matches last_valid

