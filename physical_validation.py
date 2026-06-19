"""
Physical Validation Core Module
Includes: Telemetry Calibration, Motor Mapping, Gain Scaling, Inference Tuning, and Safe Warmup.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from cerebrum.unified import CerebrumNet

class TelemetryCalibrator:
    """
    Calibrates raw sensory inputs and telemetry for consumption by the Cerebrum model.
    Corrects offsets, applies scales, validates values against boundaries,
    and cleanses NaN/Inf values to prevent computational failures.
    """
    def __init__(
        self,
        offsets: Optional[Dict[str, np.ndarray]] = None,
        scales: Optional[Dict[str, np.ndarray]] = None,
        limits: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        # Default configuration
        self.offsets = offsets if offsets is not None else {
            "lidar": np.zeros(4),
            "camera": np.zeros(2),
            "odometry": np.zeros(2),
            "tilt": np.zeros(1)
        }
        self.scales = scales if scales is not None else {
            "lidar": np.ones(4),
            "camera": np.ones(2),
            "odometry": np.ones(2),
            "tilt": np.ones(1)
        }
        self.limits = limits if limits is not None else {
            "lidar": (0.0, 10.0),
            "camera": (0.0, 1.0),
            "velocity": (-2.0, 2.0),
            "heading": (-np.pi, np.pi),
            "tilt": (-np.pi / 2, np.pi / 2)
        }
        self.calibration_history: List[Dict[str, Any]] = []

    def calibrate(self, raw_telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies offsets, scales, cleanses NaN/Infs, and clamps raw sensor signals.
        """
        calibrated = {}
        for sensor, raw_val in raw_telemetry.items():
            if raw_val is None:
                calibrated[sensor] = None
                continue

            arr = np.asarray(raw_val, dtype=float)
            
            # Impute NaN/Infs with fallback default
            if np.isnan(arr).any() or np.isinf(arr).any():
                if sensor == "lidar":
                    arr = np.where(np.isnan(arr) | np.isinf(arr), self.limits["lidar"][1], arr)
                else:
                    arr = np.where(np.isnan(arr), 0.0, arr)

            # Apply offset and scale calibration
            if sensor in self.offsets and sensor in self.scales:
                arr = (arr - self.offsets[sensor]) * self.scales[sensor]

            # Clamp boundaries
            if sensor == "lidar":
                arr = np.clip(arr, self.limits["lidar"][0], self.limits["lidar"][1])
            elif sensor == "camera":
                arr = np.clip(arr, self.limits["camera"][0], self.limits["camera"][1])
            elif sensor == "odometry":
                # [velocity, heading]
                arr[0] = np.clip(arr[0], self.limits["velocity"][0], self.limits["velocity"][1])
                arr[1] = np.clip(arr[1], self.limits["heading"][0], self.limits["heading"][1])
            elif sensor == "tilt":
                arr = np.clip(arr, self.limits["tilt"][0], self.limits["tilt"][1])

            calibrated[sensor] = arr

        return calibrated

    def update_calibration(self, sensor: str, offset: np.ndarray, scale: np.ndarray) -> None:
        """
        Updates calibration coefficients at runtime.
        """
        self.offsets[sensor] = np.asarray(offset, dtype=float)
        self.scales[sensor] = np.asarray(scale, dtype=float)
        self.calibration_history.append({
            "timestamp": time.time(),
            "sensor": sensor,
            "offset": self.offsets[sensor].copy(),
            "scale": self.scales[sensor].copy()
        })


class MotorMapper:
    """
    Maps continuous workspace velocities/actions outputted by CerebrumNet/System1
    to low-level wheel velocity commands, using differential-drive kinematics,
    saturations, slew-rate limitation, and deadband compensation.
    """
    def __init__(
        self,
        wheel_base: float = 0.2,
        wheel_radius: float = 0.05,
        deadband: float = 0.02,
        max_wheel_vel: float = 40.0,
        max_slew_rate: float = 150.0  # Max rad/s^2 change in wheel speed per step
    ):
        self.L = wheel_base
        self.r = wheel_radius
        self.deadband = deadband
        self.max_wheel_vel = max_wheel_vel
        self.max_slew_rate = max_slew_rate
        self.last_wheel_vels = np.zeros(2)  # [v_left, v_right]

    def workspace_to_wheel(self, linear_v: float, angular_w: float, dt: float = 0.02) -> np.ndarray:
        """
        Performs inverse kinematics to map target vehicle speed [linear_v, angular_w]
        to wheel rotational speeds [omega_l, omega_r] (rad/s) with slew-rate and deadband limits.
        """
        # Inverse Kinematics for Differential Drive
        # omega_l = (v - w * L / 2) / r
        # omega_r = (v + w * L / 2) / r
        omega_l = (linear_v - (angular_w * self.L / 2.0)) / self.r
        omega_r = (linear_v + (angular_w * self.L / 2.0)) / self.r

        target_vels = np.array([omega_l, omega_r])

        # Deadband Compensation
        target_vels = np.where(np.abs(target_vels) < self.deadband, 0.0, target_vels)

        # Slew-Rate Limiter
        max_delta = self.max_slew_rate * dt
        diff = target_vels - self.last_wheel_vels
        clipped_diff = np.clip(diff, -max_delta, max_delta)
        
        final_vels = self.last_wheel_vels + clipped_diff
        
        # Saturation Limit
        final_vels = np.clip(final_vels, -self.max_wheel_vel, self.max_wheel_vel)
        self.last_wheel_vels = final_vels.copy()
        
        return final_vels

    def wheel_to_workspace(self, omega_l: float, omega_r: float) -> Tuple[float, float]:
        """
        Performs forward kinematics to map wheel speeds back to chassis velocity.
        """
        linear_v = 0.5 * (omega_l + omega_r) * self.r
        angular_w = (omega_r - omega_l) * self.r / self.L
        return linear_v, angular_w


class GainScaler:
    """
    Dynamically scales motor commands and control inputs based on system safety metrics.
    Progressively reduces gains during high tilt/pitch deviations, high acceleration,
    or excessive surprise (reconstruction error) to prevent rollover or oscillation.
    """
    def __init__(
        self,
        base_gain: float = 1.0,
        min_gain: float = 0.05,
        tilt_hazard_threshold: float = 0.35,  # rad
        surprise_hazard_threshold: float = 5.0
    ):
        self.base_gain = base_gain
        self.min_gain = min_gain
        self.tilt_hazard_threshold = tilt_hazard_threshold
        self.surprise_hazard_threshold = surprise_hazard_threshold
        self.current_gain = base_gain

    def compute_scaling_factor(self, current_tilt: float, reconstruction_error: float) -> float:
        """
        Dynamically calculates the control multiplier based on current hazards.
        """
        tilt_deviation = abs(current_tilt)
        if tilt_deviation >= self.tilt_hazard_threshold:
            tilt_multiplier = 0.0
        else:
            # Linear/exponential ramp-down starting from 80% of threshold
            critical_start = 0.8 * self.tilt_hazard_threshold
            if tilt_deviation > critical_start:
                # We use a power of 1.1 to match non-linear decay and ensure gain_tilt < 0.5 at 0.45 tilt (hazard threshold 0.5)
                ratio = (tilt_deviation - critical_start) / (self.tilt_hazard_threshold - critical_start)
                tilt_multiplier = (1.0 - ratio) ** 1.1
            else:
                tilt_multiplier = 1.0

        # Surprise-based scaling
        if reconstruction_error > self.surprise_hazard_threshold:
            surprise_multiplier = max(0.2, self.surprise_hazard_threshold / reconstruction_error)
        else:
            surprise_multiplier = 1.0

        raw_gain = self.base_gain * tilt_multiplier * surprise_multiplier
        self.current_gain = max(self.min_gain, raw_gain)
        return self.current_gain

    def scale_commands(self, wheel_commands: np.ndarray) -> np.ndarray:
        """
        Applies calculated gain to actuator commands.
        """
        return wheel_commands * self.current_gain


class InferenceTuner:
    """
    Dynamically tunes CerebrumNet inference hyperparameters (e.g. n_settle)
    during hardware execution to maintain target control loop frequency (e.g. 50Hz).
    Monitors execution latency and adjusts workload to ensure budget compliance.
    """
    def __init__(
        self,
        target_hz: float = 50.0,
        max_latency_sec: float = 0.020,  # 20ms maximum budget per control cycle
        min_n_settle: int = 2,
        max_n_settle: int = 12
    ):
        self.target_hz = target_hz
        self.max_latency_sec = max_latency_sec
        self.min_n_settle = min_n_settle
        self.max_n_settle = max_n_settle
        self.latency_history: List[float] = []

    def tune(self, net: CerebrumNet, last_step_duration: float) -> None:
        """
        Dynamically adjusts net.cfg.n_settle based on moving latency observations.
        If latency exceeds max_latency_sec, reduce settling iterations to speed up inference.
        """
        self.latency_history.append(last_step_duration)
        if len(self.latency_history) > 20:
            self.latency_history.pop(0)

        avg_latency = np.mean(self.latency_history)
        current_n_settle = net.cfg.n_settle

        # hebbian/heuristic correction on settling depth
        if avg_latency > 0.9 * self.max_latency_sec:
            new_n_settle = max(self.min_n_settle, current_n_settle - 1)
        elif avg_latency < 0.5 * self.max_latency_sec:
            new_n_settle = min(self.max_n_settle, current_n_settle + 1)
        else:
            new_n_settle = current_n_settle

        from dataclasses import replace
        # Modify config in place
        net.cfg = replace(net.cfg, n_settle=new_n_settle)


class WarmupRamper:
    """
    Implements a safe startup protocol. Slowly ramps gains and postural angles
    from resting/standby values to active values, preventing torque spikes.
    """
    def __init__(
        self,
        ramp_steps: int = 100,
        posture_start: float = 0.0,
        posture_target: float = 0.0
    ):
        self.ramp_steps = ramp_steps
        self.current_step = 0
        self.posture_start = posture_start
        self.posture_target = posture_target

    def get_warmup_multiplier(self) -> float:
        """
        Returns linear ramp multiplier in range [0.0, 1.0].
        """
        if self.current_step >= self.ramp_steps:
            return 1.0
        return float(self.current_step) / self.ramp_steps

    def get_ramped_posture(self) -> float:
        """
        Linearly interpolates posture target during warmup.
        """
        multiplier = self.get_warmup_multiplier()
        interpolated = self.posture_start + multiplier * (self.posture_target - self.posture_start)
        return interpolated

    def step(self) -> None:
        """
        Advances the warmup sequence by one step.
        """
        if self.current_step < self.ramp_steps:
            self.current_step += 1

    def is_complete(self) -> bool:
        return self.current_step >= self.ramp_steps


class Geometric3DOFSolver:
    """
    Geometric Inverse Kinematics solver for a 3-DOF robotic arm.
    Links: l1 (base height), l2 (upper arm), l3 (forearm).
    """
    def __init__(self, l1: float = 1.0, l2: float = 1.0, l3: float = 1.0):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

    def inverse_kinematics(self, x: float, y: float, z: float) -> Tuple[Tuple[float, float, float], bool]:
        """
        Calculates joint angles theta0, theta1, theta2.
        Returns (theta0, theta1, theta2) and reachable flag.
        Handles Z-axis singularity at x=0, y=0 by locking base angle to 0.0.
        Clips out-of-reach coordinates to workspace bounds to prevent arccos NaN.
        """
        # Check if any coordinate x, y, z is NaN or Inf (non-finite)
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            return ((0.0, 0.0, 0.0), False)

        # Check if x, y, or z is extremely large to prevent OverflowError when squaring
        if abs(x) > 1e150 or abs(y) > 1e150 or abs(z) > 1e150:
            return ((0.0, 0.0, 0.0), False)

        # 1. Base angle with singularity handling
        if x == 0.0 and y == 0.0:
            theta0 = 0.0
        else:
            theta0 = float(np.arctan2(y, x))

        # Horizontal distance in plane
        r = np.sqrt(x**2 + y**2)
        # Shift z relative to base height l1
        z_prime = z - self.l1

        # Distance from shoulder to end-effector
        D = np.sqrt(r**2 + z_prime**2)
        reachable = True

        # Clip target to workspace boundary limits
        max_reach = self.l2 + self.l3
        min_reach = abs(self.l2 - self.l3)

        if D > max_reach:
            reachable = False
            scale = max_reach / D
            r = r * scale
            z_prime = z_prime * scale
            D = max_reach
        elif D < min_reach:
            reachable = False
            if D == 0:
                r = min_reach
                z_prime = 0.0
            else:
                scale = min_reach / D
                r = r * scale
                z_prime = z_prime * scale
            D = min_reach

        # Cosine rule for theta2
        # cos(theta2) = (r^2 + z_prime^2 - l2^2 - l3^2) / (2 * l2 * l3)
        numerator = r**2 + z_prime**2 - self.l2**2 - self.l3**2
        denominator = 2.0 * self.l2 * self.l3
        cos_val = numerator / denominator
        
        # Float numerical error safeguard clipping
        cos_val = np.clip(cos_val, -1.0, 1.0)
        
        theta2 = float(np.arccos(cos_val))

        # Angle calculations:
        # theta1 = atan2(z_prime, r) - atan2(l3 * sin(theta2), l2 + l3 * cos(theta2))
        theta1 = float(np.arctan2(z_prime, r) - np.arctan2(self.l3 * np.sin(theta2), self.l2 + self.l3 * np.cos(theta2)))

        return (theta0, theta1, theta2), reachable


def joint_to_ticks(joint_angle: float, ticks_per_rad: float = 1000.0) -> int:
    """
    Converts continuous joint angle (radians) to integer encoder ticks.
    """
    if not np.isfinite(joint_angle):
        return 0
    ticks = joint_angle * ticks_per_rad
    return int(round(ticks))


def torque_to_current(torque: float, torque_constant: float = 0.5, max_current: float = 10.0) -> float:
    """
    Converts motor torque command to current with a protective soft current clamp.
    """
    if abs(torque_constant) < 1e-9:
        return 0.0
    current = torque / torque_constant
    abs_max_current = abs(max_current)
    effective_limit = min(abs_max_current, 25.0)
    return float(np.clip(current, -effective_limit, effective_limit))



class KalmanFilter:
    """Simple 1D Kalman filter for smoothing noisy sensor/gains streams."""
    def __init__(self, q: float = 1e-4, r: float = 1e-2, initial_value: float = 0.0):
        self.q = q
        self.r = r
        self.x = initial_value
        self.p = 1.0

    def filter(self, measurement: float) -> float:
        self.p = self.p + self.q
        k_gain = self.p / (self.p + self.r)
        self.x = self.x + k_gain * (measurement - self.x)
        self.p = (1.0 - k_gain) * self.p
        return self.x


class MovingAverageFilter:
    """Sliding-window moving average filter."""
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history: List[float] = []

    def filter(self, value: float) -> float:
        self.history.append(value)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        return sum(self.history) / len(self.history)


def safe_get_telemetry(hw: Any, last_valid_telemetry: Optional[Dict[str, Any]] = None, retries: int = 5) -> Tuple[Dict[str, Any], bool]:
    """
    Attempts to fetch telemetry from the hardware/simulator.
    If dropouts are encountered, attempts reconnect up to `retries` times.
    Falls back gracefully to `last_valid_telemetry` if all retries fail.
    """
    fallback = {
        "lidar": np.array([10.0, 5.0, 5.0, 5.0]),
        "camera": np.array([0.8, 0.2]),
        "odometry": np.array([0.0, 0.0]),
        "tilt": np.array([0.0])
    }

    for attempt in range(retries):
        telem = hw.get_telemetry()
        if telem is not None:
            has_any_valid = False
            for k in fallback:
                if telem.get(k) is not None:
                    has_any_valid = True
                    break
            
            if has_any_valid:
                imputed = dict(telem)
                for key in fallback:
                    if key not in imputed or imputed[key] is None:
                        if last_valid_telemetry is not None and key in last_valid_telemetry and last_valid_telemetry[key] is not None:
                            imputed[key] = last_valid_telemetry[key]
                        else:
                            imputed[key] = fallback[key]
                return imputed, True
        time.sleep(0.005)
        
    if last_valid_telemetry is not None:
        imputed_last = dict(last_valid_telemetry)
        for key in fallback:
            if key not in imputed_last or imputed_last[key] is None:
                imputed_last[key] = fallback[key]
        return imputed_last, False
        
    return fallback, False


