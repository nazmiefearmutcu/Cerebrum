"""
Standalone Verification Runner for Sim2Real Physical Validation
Runs Pre-Run Diagnostics, Kinematics Verification, Dynamics Verification, and Hebbian updates verification.
"""

import json
import time
import numpy as np
import torch
from typing import Dict, Any
from cerebrum.unified import CerebrumNet
from cerebrum.config import CerebrumConfig
from cerebrum.types import Exogenous
from cerebrum.grounding import System1Reflex
from physical_validation import (
    TelemetryCalibrator, MotorMapper, GainScaler
)

class MockHardware:
    """
    Simulates the physical dynamics, kinematics, sensor noise, bias, and dropouts
    of a differential-drive ground robot.
    """
    def __init__(
        self,
        wheel_base: float = 0.2,
        wheel_radius: float = 0.05,
        time_step: float = 0.020,  # 50Hz control cycle
        motor_tau: float = 0.15,   # Motor time constant in seconds
        noise_std_lidar: float = 0.015,
        noise_std_imu: float = 0.02,
        gyro_drift_rate: float = 0.001,
        dropout_probability: float = 0.05
    ):
        self.L = wheel_base
        self.r = wheel_radius
        self.dt = time_step
        self.tau = motor_tau
        
        # Noise settings
        self.noise_std_lidar = noise_std_lidar
        self.noise_std_imu = noise_std_imu
        self.drift_rate = gyro_drift_rate
        self.p_dropout = dropout_probability
        
        # State variables
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v_l = 0.0  # Actual left wheel angular velocity (rad/s)
        self.v_r = 0.0  # Actual right wheel angular velocity (rad/s)
        self.tilt = 0.0  # Vehicle pitch angle (rad)
        self.gyro_drift = 0.0
        
        # Motor Targets
        self.target_v_l = 0.0
        self.target_v_r = 0.0

    def set_motor_commands(self, left_rad_s: float, right_rad_s: float) -> None:
        """
        Receives actuator commands from controller. Simulates packet loss.
        """
        # Dropout: If triggered, command is lost (previous command remains active: Zero-Order Hold)
        if np.random.random() >= self.p_dropout:
            self.target_v_l = left_rad_s
            self.target_v_r = right_rad_s

    def step(self, external_tilt_dist: float = 0.0) -> None:
        """
        Advances the physics simulator by dt.
        """
        # First-order motor response lag dynamics
        self.v_l += (self.dt / self.tau) * (self.target_v_l - self.v_l)
        self.v_r += (self.dt / self.tau) * (self.target_v_r - self.v_r)
        
        # Differential kinematics
        v_linear = 0.5 * (self.v_l + self.v_r) * self.r
        omega = (self.v_r - self.v_l) * self.r / self.L
        
        # Integration
        self.yaw += omega * self.dt
        self.x += v_linear * np.cos(self.yaw) * self.dt
        self.y += v_linear * np.sin(self.yaw) * self.dt
        
        # Heading gyroscope random walk drift simulation
        self.gyro_drift += np.random.normal(0.0, self.drift_rate * self.dt)
        
        # Pitch / tilt simulation based on acceleration and external inclination
        accel = (v_linear - 0.5 * (self.v_l + self.v_r) * self.r) / self.dt
        self.tilt = external_tilt_dist + 0.05 * accel

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Generates sensory outputs containing simulation states, noise, drift, and possible NaNs.
        """
        # Random telemetry sensor dropout
        if np.random.random() < self.p_dropout:
            return {
                "lidar": None,
                "camera": None,
                "odometry": None,
                "tilt": None
            }

        # Lidar ranges (e.g. 4 directions: [front, left, rear, right])
        # Base obstacle at (x=2.0, y=0.0). Sensor reads Euclidean distance.
        dist_to_wall = max(0.05, 2.0 - self.x)
        lidar_data = np.array([dist_to_wall, 5.0, 5.0, 5.0]) + np.random.normal(0.0, self.noise_std_lidar, 4)
        
        # Camera simulation
        camera_data = np.array([0.8, 0.2]) + np.random.normal(0.0, 0.02, 2)
        
        # Odometry: [measured_velocity, measured_heading]
        measured_vel = 0.5 * (self.v_l + self.v_r) * self.r + np.random.normal(0.0, 0.01)
        measured_heading = self.yaw + self.gyro_drift + np.random.normal(0.0, self.noise_std_imu)
        odometry_data = np.array([measured_vel, measured_heading])
        
        # Tilt sensor (IMU)
        measured_tilt = np.array([self.tilt + np.random.normal(0.0, self.noise_std_imu)])
        
        return {
            "lidar": lidar_data,
            "camera": camera_data,
            "odometry": odometry_data,
            "tilt": measured_tilt
        }


def run_checklist(hw: MockHardware, calibrator: TelemetryCalibrator, net: CerebrumNet) -> Dict[str, Any]:
    """
    Executes Phase 1: Pre-Run Diagnostic Checklist.
    """
    results = {"status": "PASS", "errors": []}
    
    # 1. Device check
    if not hasattr(net, "device"):
        results["status"] = "FAIL"
        results["errors"].append("CerebrumNet device property missing.")
        
    # 2. Telemetry validation ping (retry to handle dropouts during diagnostic check)
    telem = None
    for _ in range(10):
        t = hw.get_telemetry()
        if t is not None and t["lidar"] is not None and t["camera"] is not None:
            telem = t
            break
            
    if telem is None:
        results["status"] = "FAIL"
        results["errors"].append("Critical communication loss: all sensory topics are offline.")
        return results
    
    # 3. Voltage verification (simulated nominal > 11.1V)
    battery_voltage = 12.0
    if battery_voltage < 11.1:
        results["status"] = "FAIL"
        results["errors"].append(f"Low voltage: {battery_voltage}V (limit 11.1V)")

    # 4. Check for initial NaN corruption
    for k, val in telem.items():
        if val is not None and np.isnan(val).any():
            results["status"] = "FAIL"
            results["errors"].append(f"NaN elements found on uncalibrated topic: {k}")
            
    return results


def run_kinematics_verification(hw: MockHardware, mapper: MotorMapper, calibrator: TelemetryCalibrator) -> Dict[str, Any]:
    """
    Executes Phase 2: Kinematics Verification.
    """
    results = {"status": "PASS", "errors": []}
    
    # Temporarily disable dropouts to ensure clean measurement
    old_p_dropout = hw.p_dropout
    hw.p_dropout = 0.0
    
    # Verify forward movement kinematics
    target_v, target_w = 0.5, 0.0
    
    # Reset kinematics state
    mapper.last_wheel_vels = np.zeros(2)
    hw.v_l = 0.0
    hw.v_r = 0.0
    hw.target_v_l = 0.0
    hw.target_v_r = 0.0
    
    # Step simulation enough times to bypass transient motor lag and slew-rate limits
    # Target speed is 10.0 rad/s. Max slew rate is 150.0 rad/s^2.
    # Acceleration takes 10.0 / 150.0 = 0.067s. Motor lag tau = 0.15s.
    # 150 steps = 3.0s is way more than enough to achieve steady state.
    for _ in range(150):
        wheel_cmds = mapper.workspace_to_wheel(target_v, target_w, dt=hw.dt)
        hw.set_motor_commands(wheel_cmds[0], wheel_cmds[1])
        hw.step()
        
    # Read kinematic output
    measured_v, measured_w = mapper.wheel_to_workspace(hw.v_l, hw.v_r)
    
    v_error = abs(measured_v - target_v)
    w_error = abs(measured_w - target_w)
    
    if v_error > 0.05:
        results["status"] = "FAIL"
        results["errors"].append(f"Forward velocity error exceeds threshold: {v_error:.4f} > 0.05")
    if w_error > 0.02:
        results["status"] = "FAIL"
        results["errors"].append(f"Angular rate drift exceeds threshold: {w_error:.4f} > 0.02")
        
    hw.p_dropout = old_p_dropout
    return results


def run_dynamics_verification(
    hw: MockHardware,
    mapper: MotorMapper,
    calibrator: TelemetryCalibrator,
    scaler: GainScaler,
    reflex: System1Reflex
) -> Dict[str, Any]:
    """
    Executes Phase 3: Dynamics Verification.
    """
    results = {"status": "PASS", "errors": []}
    
    # Temporarily disable dropouts to ensure clean measurement
    old_p_dropout = hw.p_dropout
    hw.p_dropout = 0.0
    
    # 1. Step Response transient profile verification
    target_vel = 1.5
    
    # Reset kinematics state
    mapper.last_wheel_vels = np.zeros(2)
    hw.v_l = 0.0
    hw.v_r = 0.0
    hw.target_v_l = 0.0
    hw.target_v_r = 0.0
    
    rise_time_steps = 0
    steady_state_achieved = False
    
    for _ in range(100):
        wheel_cmds = mapper.workspace_to_wheel(target_vel, 0.0, dt=hw.dt)
        hw.set_motor_commands(wheel_cmds[0], wheel_cmds[1])
        hw.step()
        measured_v, _ = mapper.wheel_to_workspace(hw.v_l, hw.v_r)
        if measured_v < 0.9 * target_vel and not steady_state_achieved:
            rise_time_steps += 1
        else:
            steady_state_achieved = True

    rise_time_sec = rise_time_steps * hw.dt
    if rise_time_sec > 0.6:  # Limits rise response limits
        results["status"] = "FAIL"
        results["errors"].append(f"Response too sluggish: motor rise time {rise_time_sec:.3f}s exceeds 0.6s")

    # Restore dropout configuration
    hw.p_dropout = old_p_dropout

    # 2. Reflex stability (System 1) trigger verification under tilt hazard
    # Reset pose and wheel speeds so that the robot is far from the wall and collision reflex doesn't override the tilt reflex
    hw.x = 0.0
    hw.y = 0.0
    hw.yaw = 0.0
    hw.v_l = 0.0
    hw.v_r = 0.0
    hw.target_v_l = 0.0
    hw.target_v_r = 0.0
    hw.gyro_drift = 0.0
    
    # Inject tilt deviation manually representing imbalance
    hw.step(external_tilt_dist=0.6)  # Exceeds reflex's tilt threshold (0.5)
    telem = hw.get_telemetry()
    
    # Retrieve valid telemetry (retry to bypass random packet loss)
    for _ in range(10):
        if telem is not None and telem["lidar"] is not None and telem["tilt"] is not None:
            break
        telem = hw.get_telemetry()
        
    if telem is None or telem["lidar"] is None or telem["tilt"] is None:
        results["status"] = "FAIL"
        results["errors"].append("Failed to retrieve telemetry during reflex test.")
        return results

    calibrated = calibrator.calibrate(telem)
    
    # Evaluate System 1 bypass
    triggered, bypass_cmd = reflex.evaluate(
        {"dist": float(calibrated["lidar"][0]), "tilt": float(calibrated["tilt"][0]), "error_energy": 0.0}
    )
    
    if not triggered:
        results["status"] = "FAIL"
        results["errors"].append("Tilt threshold exceeded but System 1 reflex failed to activate.")
    else:
        # Check command is stabilizing (backward or safety brakes)
        if not np.array_equal(bypass_cmd, np.array([-1.0, -1.0])):
            results["status"] = "FAIL"
            results["errors"].append(f"Incorrect reflex stabilization command: {bypass_cmd}")
            
    # Reset simulation inclination
    hw.tilt = 0.0
    hw.step()
    return results


def get_weights_numpy(tensor_wrapper):
    if hasattr(tensor_wrapper, "_tensor"):
        return tensor_wrapper._tensor.detach().cpu().numpy().copy()
    elif isinstance(tensor_wrapper, torch.Tensor):
        return tensor_wrapper.detach().cpu().numpy().copy()
    else:
        return np.array(tensor_wrapper).copy()


def run_hebbian_verification(net: CerebrumNet) -> Dict[str, Any]:
    """
    Executes Phase 4: Hebbian Updates and Metaplastic Fuse Gating verification.
    """
    results = {"status": "PASS", "errors": []}
    
    # Save target parameters initial state
    mod = net.modules[0]
    W_init = get_weights_numpy(mod.W[0])
    
    # 1. Verify plasticity is functional (dW != 0 under reward)
    obs = [np.ones(net.slice_dim) * 0.5 for _ in range(net.M_)]
    action = Exogenous(np.array([0.1, 0.0]))
    
    # Temporarily force open the metaplastic fuses to measure maximum plasticity potential
    net._force_theta = 1.0
    z, M = net.step(obs, action=action, reward=1.0)
    
    W_step1 = get_weights_numpy(mod.W[0])
    diff_unfused = np.sum(np.abs(W_step1 - W_init))
    
    if diff_unfused == 0.0:
        results["status"] = "FAIL"
        results["errors"].append("Hebbian learning is inactive: weight change was exactly 0.0")

    # 2. Verify Metaplastic fuse gating constraint (theta=0 blocks update)
    net._force_theta = 0.0
    _ = net.step(obs, action=action, reward=1.0)
    
    W_step2 = get_weights_numpy(mod.W[0])
    diff_fused = np.sum(np.abs(W_step2 - W_step1))
    
    if diff_fused > 1e-12:
        results["status"] = "FAIL"
        results["errors"].append(f"Metaplastic fuse leakage: weights updated by {diff_fused:.4e} with theta forced to 0.0")

    # Clean up test hooks
    net._force_theta = None
    return results


def main():
    # Setup network config
    cfg = CerebrumConfig(dims=(4, 8), n_settle=6, seed=42)
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg)
    
    # Setup verification components
    hw = MockHardware()
    calibrator = TelemetryCalibrator()
    mapper = MotorMapper()
    scaler = GainScaler()
    reflex = System1Reflex(tilt_threshold=0.5)
    
    print("Beginning Physical Validation Protocol...")
    logs = {"timestamp": time.time(), "phases": {}}
    
    # Phase 1
    logs["phases"]["checklist"] = run_checklist(hw, calibrator, net)
    print(f"Checklist: {logs['phases']['checklist']['status']}")
    
    # Phase 2
    logs["phases"]["kinematics"] = run_kinematics_verification(hw, mapper, calibrator)
    print(f"Kinematics Verification: {logs['phases']['kinematics']['status']}")
    
    # Phase 3
    logs["phases"]["dynamics"] = run_dynamics_verification(hw, mapper, calibrator, scaler, reflex)
    print(f"Dynamics Verification: {logs['phases']['dynamics']['status']}")
    
    # Phase 4
    logs["phases"]["hebbian"] = run_hebbian_verification(net)
    print(f"Hebbian Updates Verification: {logs['phases']['hebbian']['status']}")
    
    # Global state evaluation
    logs["overall_status"] = "PASS" if all(phase["status"] == "PASS" for phase in logs["phases"].values()) else "FAIL"
    print(f"Global Validation Status: {logs['overall_status']}")
    
    # Write report
    with open("physical_validation_logs.json", "w") as f:
        json.dump(logs, f, indent=4)
    print("Logs successfully dumped to physical_validation_logs.json")

if __name__ == "__main__":
    main()
