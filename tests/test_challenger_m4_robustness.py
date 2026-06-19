import pytest
import numpy as np
import math
from power_parser import KalmanFilter as PowerKalmanFilter, MovingAverageFilter as PowerMovingAverageFilter
from physical_validation import (
    KalmanFilter as PhysKalmanFilter,
    MovingAverageFilter as PhysMovingAverageFilter,
    TelemetryCalibrator,
    MotorMapper
)

# =====================================================================
# 1. Filter Convergence & Transient Step Responses
# =====================================================================

def test_moving_average_step_convergence():
    """
    Verify convergence times and behaviors for Moving Average filters
    under step-changes in power draw (e.g., stepping from 4.2W to 25.0W).
    """
    # Test PowerMovingAverageFilter
    ma_power = PowerMovingAverageFilter(window_size=5)
    
    # Initialize history to baseline 4.2W
    for _ in range(5):
        ma_power.filter(4.2)
    assert ma_power.filter(4.2) == pytest.approx(4.2)
    
    # Step change to 25.0W
    outputs = []
    for _ in range(6):
        outputs.append(ma_power.filter(25.0))
        
    # Check transient step behavior
    assert outputs[0] == pytest.approx((4.2 * 4 + 25.0) / 5) # 8.36
    assert outputs[1] == pytest.approx((4.2 * 3 + 25.0 * 2) / 5) # 12.52
    assert outputs[2] == pytest.approx((4.2 * 2 + 25.0 * 3) / 5) # 16.68
    assert outputs[3] == pytest.approx((4.2 * 1 + 25.0 * 4) / 5) # 20.84
    assert outputs[4] == pytest.approx(25.0) # Full convergence in exactly N steps
    assert outputs[5] == pytest.approx(25.0) # Maintains target

    # Test PhysMovingAverageFilter
    ma_phys = PhysMovingAverageFilter(window_size=3)
    for _ in range(3):
        ma_phys.filter(4.2)
    assert ma_phys.filter(4.2) == pytest.approx(4.2)
    
    outputs_phys = [ma_phys.filter(25.0) for _ in range(4)]
    assert outputs_phys[0] == pytest.approx((4.2 * 2 + 25.0) / 3)
    assert outputs_phys[2] == pytest.approx(25.0) # Converges in 3 steps


def test_kalman_step_convergence():
    """
    Verify convergence times and behaviors for Kalman filters under step-changes
    in power draw (e.g., stepping from 4.2W to 25.0W).
    """
    # Kalman filter parameters: q (process noise), r (measurement noise)
    # Target value: 25.0W, initial value: 4.2W
    
    # Scenario A: Standard Kalman Filter in power_parser.py
    kf_power = PowerKalmanFilter(q=1e-4, r=1e-2, initial_value=4.2)
    
    # Pre-warm the filter so that error covariance P converges to its steady-state value
    for _ in range(50):
        kf_power.filter(4.2, dt=1.0)
    
    # Apply step-change to 25.0W
    steps = 0
    val = 4.2
    limit = 25.0 - 0.05 * (25.0 - 4.2) # 95% convergence threshold = 23.96W
    
    outputs = []
    for _ in range(100):
        val = kf_power.filter(25.0, dt=1.0)
        outputs.append(val)
        steps += 1
        if val >= limit:
            break
            
    # With small q (1e-4) relative to r (1e-2), convergence is slow (takes 30 steps)
    assert steps > 5
    assert val == pytest.approx(25.0, abs=1.5)
    
    # Scenario B: Impact of larger process noise q (trusts measurement more)
    kf_fast = PowerKalmanFilter(q=1e-2, r=1e-2, initial_value=4.2)
    # Pre-warm
    for _ in range(50):
        kf_fast.filter(4.2, dt=1.0)
        
    steps_fast = 0
    val_fast = 4.2
    for _ in range(100):
        val_fast = kf_fast.filter(25.0, dt=1.0)
        steps_fast += 1
        if val_fast >= limit:
            break
            
    # Larger q must converge faster
    assert steps_fast < steps
    
    # Scenario C: Impact of dt on power_parser.py Kalman filter
    # For a larger dt, process noise covariance self.p = self.p + self.q * dt is larger.
    # Therefore, Kalman gain is larger, and convergence per step is faster.
    kf_dt1 = PowerKalmanFilter(q=1e-3, r=1e-2, initial_value=4.2)
    kf_dt5 = PowerKalmanFilter(q=1e-3, r=1e-2, initial_value=4.2)
    
    val_dt1 = kf_dt1.filter(25.0, dt=1.0)
    val_dt5 = kf_dt5.filter(25.0, dt=5.0)
    
    # Larger dt must cause larger single-step adjustment
    assert val_dt5 > val_dt1
    assert val_dt5 <= 25.0

    # Scenario D: Physical validation Kalman Filter (which has no dt scaling)
    kf_phys = PhysKalmanFilter(q=1e-4, r=1e-2, initial_value=4.2)
    val_phys = kf_phys.filter(25.0)
    assert val_phys > 4.2
    assert val_phys < 25.0


def test_kalman_filter_bounds_and_anomalies():
    """
    Test Kalman Filter behavior under anomalous state variables (e.g. dt=0, dt<0).
    """
    kf = PowerKalmanFilter(q=1e-4, r=1e-2, initial_value=4.2)
    
    # Zero dt: self.p remains unchanged in prediction step
    p_before = kf.p
    res = kf.filter(25.0, dt=0.0)
    assert kf.p <= p_before # error covariance should reduce or stay same after measurement update
    assert res > 4.2
    
    # Negative dt: self.p decreases in prediction step
    kf_neg = PowerKalmanFilter(q=1e-4, r=1e-2, initial_value=4.2)
    res_neg = kf_neg.filter(25.0, dt=-10.0)
    assert not np.isnan(res_neg)
    assert not np.isinf(res_neg)


# =====================================================================
# 2. Telemetry Calibrator Boundary Clamping under NaN/Inf limits
# =====================================================================

def test_telemetry_calibrator_anomalous_limits():
    """
    Verify telemetry calibrator boundary clamping behavior when configured
    limits contain NaN or Inf values.
    """
    # 1. Configured limit with NaN: limits={"lidar": (np.nan, 10.0)}
    calibrator_nan = TelemetryCalibrator(
        limits={
            "lidar": (np.nan, 10.0),
            "camera": (0.0, 1.0),
            "velocity": (-2.0, 2.0),
            "heading": (-np.pi, np.pi),
            "tilt": (-np.pi / 2, np.pi / 2)
        }
    )
    
    raw_telem = {
        "lidar": np.array([2.0, 5.0, 8.0, 11.0]),
        "camera": np.array([0.5, 0.5])
    }
    
    # When np.clip is run with np.nan as min limit:
    # np.clip(arr, np.nan, 10.0) returns [nan, nan, nan, nan] in numpy!
    calibrated_nan = calibrator_nan.calibrate(raw_telem)
    
    # Verify that the array becomes NaN, confirming numpy's clipping behavior with NaN limits
    assert np.isnan(calibrated_nan["lidar"]).all()
    
    # 2. Configured limit with Inf: limits={"camera": (0.0, np.inf)}
    calibrator_inf = TelemetryCalibrator(
        limits={
            "lidar": (0.0, 10.0),
            "camera": (0.0, np.inf),
            "velocity": (-2.0, 2.0),
            "heading": (-np.pi, np.pi),
            "tilt": (-np.pi / 2, np.pi / 2)
        }
    )
    
    # camera shape is 2, so raw input must be size 2
    raw_telem_inf = {
        "camera": np.array([9999.0, np.inf])
    }
    
    calibrated_inf = calibrator_inf.calibrate(raw_telem_inf)
    
    # Clamping with np.inf should allow large values and inf to pass through unchanged
    assert calibrated_inf["camera"][0] == 9999.0
    assert calibrated_inf["camera"][1] == np.inf
    
    # 3. Input telemetry contains NaN/Inf with normal limits
    # Confirm that inputs with NaN/Inf are handled by TelemetryCalibrator
    calibrator_normal = TelemetryCalibrator()
    raw_bad_inputs = {
        "lidar": np.array([np.nan, np.inf, -np.inf, 5.0]),
        "camera": np.array([np.nan, np.inf])
    }
    
    calibrated_normal = calibrator_normal.calibrate(raw_bad_inputs)
    # Lidar is imputed to max limit (10.0) for both positive and negative NaN/Inf, and clipped
    assert np.all(calibrated_normal["lidar"] == np.array([10.0, 10.0, 10.0, 5.0]))
    # Camera is imputed to 0.0 for NaN, and Inf is clipped to max limit (1.0)
    assert np.all(calibrated_normal["camera"] == np.array([0.0, 1.0]))


def test_telemetry_calibrator_shape_mismatch():
    """
    Verify that TelemetryCalibrator raises ValueError on input shape mismatch,
    confirming lack of input shape validation.
    """
    calibrator = TelemetryCalibrator()
    raw_bad_shape = {
        "camera": np.array([0.5, 0.5, 0.5]) # camera expects size 2
    }
    with pytest.raises(ValueError, match="operands could not be broadcast together"):
        calibrator.calibrate(raw_bad_shape)


# =====================================================================
# 3. Actuator Saturation Clipping & Motor Mapping
# =====================================================================

def test_motor_mapper_saturation_boundaries():
    """
    Verify actuator saturation clipping and boundary behavior in motor mappings.
    Checks maximum velocity clamping, slew-rate limits, and deadband behavior.
    """
    # MotorMapper parameters: max_wheel_vel=40.0, max_slew_rate=150.0, deadband=0.02
    mapper = MotorMapper(wheel_base=0.2, wheel_radius=0.05, deadband=0.02, max_wheel_vel=40.0, max_slew_rate=150.0)
    
    # 1. Verify standard kinematics mapping:
    # omega_l = (v - w * L / 2) / r
    # With v = 1.0, w = 0.0: omega_l = 1.0 / 0.05 = 20.0 rad/s
    # With v = 1.0, w = 2.0: omega_l = (1.0 - 2.0 * 0.2 / 2) / 0.05 = (1.0 - 0.2) / 0.05 = 16.0 rad/s
    #                        omega_r = (1.0 + 2.0 * 0.2 / 2) / 0.05 = 1.2 / 0.05 = 24.0 rad/s
    
    # Let's bypass slew rate by using a huge dt
    vels = mapper.workspace_to_wheel(1.0, 2.0, dt=10.0)
    assert vels[0] == pytest.approx(16.0)
    assert vels[1] == pytest.approx(24.0)
    
    # 2. Saturation limit clamping:
    # Target values that exceed max_wheel_vel = 40.0
    # v = 5.0, w = 0.0 -> omega = 5.0 / 0.05 = 100.0 rad/s
    vels_sat = mapper.workspace_to_wheel(5.0, 0.0, dt=10.0)
    assert np.all(vels_sat == 40.0)
    
    # Target values that are negative and exceed limit
    vels_sat_neg = mapper.workspace_to_wheel(-5.0, 0.0, dt=10.0)
    assert np.all(vels_sat_neg == -40.0)
    
    # Reset internal state to zero
    mapper.last_wheel_vels = np.zeros(2)
    
    # 3. Slew-rate limitation at boundaries:
    # Max acceleration: max_slew_rate = 150.0 rad/s^2.
    # At dt = 0.02, max change is 150.0 * 0.02 = 3.0 rad/s.
    # If target is 20.0 rad/s starting from 0.0, first output must be 3.0 rad/s.
    vels_slew_1 = mapper.workspace_to_wheel(1.0, 0.0, dt=0.02)
    assert np.all(vels_slew_1 == 3.0)
    
    # Second step target still 20.0, output should rise to 6.0 rad/s
    vels_slew_2 = mapper.workspace_to_wheel(1.0, 0.0, dt=0.02)
    assert np.all(vels_slew_2 == 6.0)
    
    # 4. Deadband boundary behavior:
    # If target wheel velocity is below deadband (0.02), it is set to 0.0.
    # Target wheel velocity = 0.015 rad/s -> v = 0.015 * 0.05 = 0.00075 m/s
    mapper.last_wheel_vels = np.zeros(2)
    vels_dead = mapper.workspace_to_wheel(0.00075, 0.0, dt=10.0)
    assert np.all(vels_dead == 0.0)
    
    # Just above deadband: target = 0.025 rad/s -> v = 0.025 * 0.05 = 0.00125 m/s
    vels_above_dead = mapper.workspace_to_wheel(0.00125, 0.0, dt=10.0)
    assert np.allclose(vels_above_dead, 0.025)


def test_motor_mapper_state_contamination():
    """
    Verify state contamination/pollution in MotorMapper when fed with NaN/Inf values.
    """
    mapper = MotorMapper(max_wheel_vel=40.0)
    
    # Feed NaN values to workspace velocities
    vel_nan = mapper.workspace_to_wheel(float('nan'), 0.0, dt=0.02)
    assert np.isnan(vel_nan).all()
    
    # Verify that the internal state `last_wheel_vels` is now contaminated with NaNs
    assert np.isnan(mapper.last_wheel_vels).all()
    
    # Feed valid inputs, and verify that the output remains NaN because the internal state was contaminated
    vel_after = mapper.workspace_to_wheel(1.0, 0.0, dt=0.02)
    assert np.isnan(vel_after).all()
    
    # Re-initialize to test Inf behavior
    mapper_inf = MotorMapper(max_wheel_vel=40.0, max_slew_rate=150.0)
    
    # Feed Inf values
    # Because of slew rate limiter, target_vels becomes [inf, inf], diff = [inf, inf],
    # but np.clip(diff, -max_delta, max_delta) clamps it to max_delta!
    # So the output is clipped, and internal state does NOT jump to Inf immediately.
    # Let's verify:
    vel_inf = mapper_inf.workspace_to_wheel(float('inf'), 0.0, dt=0.02)
    max_delta = 150.0 * 0.02 # 3.0
    assert vel_inf[0] == pytest.approx(max_delta)
    assert vel_inf[1] == pytest.approx(max_delta)
    assert not np.isinf(mapper_inf.last_wheel_vels).any()
    
    # But if dt = float('inf') or max_slew_rate = float('inf'), max_delta is inf.
    # Let's see what happens:
    mapper_unbounded = MotorMapper(max_wheel_vel=40.0, max_slew_rate=float('inf'))
    vel_unbounded = mapper_unbounded.workspace_to_wheel(float('inf'), 0.0, dt=1.0)
    # target is inf, slew limit is inf, so final_vels is inf, but then it's clipped to max_wheel_vel (40.0)
    assert vel_unbounded[0] == 40.0
    assert vel_unbounded[1] == 40.0
    assert mapper_unbounded.last_wheel_vels[0] == 40.0
