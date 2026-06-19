import numpy as np
from typing import Dict, List, Tuple, Any

class CognitiveBridge:
    """
    Bridge connecting low-level robot telemetry to CerebrumNet predictive coding modules.
    """
    def __init__(self):
        # Identity generative matrices
        self.W0 = np.identity(5, dtype=float)
        self.W1 = np.identity(5, dtype=float)
        self.W2 = np.identity(5, dtype=float)
        
        # State vectors
        self.x0 = np.zeros(5, dtype=float)
        self.x1 = np.zeros(5, dtype=float)
        self.x2 = np.zeros(5, dtype=float)
        
        # Precision and learning rates
        self.alpha = 0.01
        self.gamma = 0.1
        
        self.observations = [np.zeros(5, dtype=float), np.zeros(5, dtype=float), np.zeros(5, dtype=float)]

    def reset_network(self) -> None:
        """Resets state vectors to zeros."""
        self.x0 = np.zeros(5, dtype=float)
        self.x1 = np.zeros(5, dtype=float)
        self.x2 = np.zeros(5, dtype=float)

    def map_telemetry_to_observations(self, raw_telemetry: Dict[str, Any]) -> List[np.ndarray]:
        """
        Validates, filters, and maps raw telemetry into partitioned observation slices.
        """
        if not isinstance(raw_telemetry, dict) or not raw_telemetry:
            raise ValueError("Telemetry data must be a non-empty dictionary.")

        required_keys = {"ik_error", "wheel_slip", "fluid_slosh", "g_force"}
        for key in required_keys:
            if key not in raw_telemetry:
                raise ValueError(f"Missing required telemetry key: '{key}'")

        # Validation for type, non-finite values, and collection types
        validated = {}
        for key in required_keys:
            val = raw_telemetry[key]
            if val is None:
                raise ValueError(f"Telemetry key '{key}' value is None.")
            
            # Check if value is a collection
            if isinstance(val, (list, dict, np.ndarray, tuple)):
                raise ValueError(f"Telemetry key '{key}' value cannot be a collection.")
            
            try:
                f_val = float(val)
            except (TypeError, ValueError):
                raise ValueError(f"Telemetry key '{key}' value cannot be cast to float.")
                
            if not np.isfinite(f_val):
                raise ValueError(f"Telemetry key '{key}' has non-finite value: {f_val}")
                
            validated[key] = f_val

        # Clip values to range [-1.0, 1.0]
        ik_err = np.clip(validated["ik_error"], -1.0, 1.0)
        slip = np.clip(validated["wheel_slip"], -1.0, 1.0)
        slosh = np.clip(validated["fluid_slosh"], -1.0, 1.0)
        g_force = np.clip(validated["g_force"], -1.0, 1.0)

        # Slice 1: Kinematics
        slice0 = np.array([ik_err, 0.0, 0.0, 0.0, 0.0], dtype=float)
        # Slice 2: Dynamics & Traction
        slice1 = np.array([slip, slosh, 0.0, 0.0, 0.0], dtype=float)
        # Slice 3: Balance & Landing
        slice2 = np.array([g_force, 0.0, 0.0, 0.0, 0.0], dtype=float)

        return [slice0, slice1, slice2]

    def get_free_energy(self) -> float:
        """
        Computes the total free energy functional across all active modules.
        """
        # epsilon_{o, i} = o_i - W_i @ x_i
        # epsilon_{x, i} = x_i
        # F_i = 0.5 * ||epsilon_{o, i}||_2^2 + 0.5 * alpha * ||epsilon_{x, i}||_2^2
        
        eps_o0 = self.observations[0] - self.W0 @ self.x0
        eps_x0 = self.x0
        F0 = 0.5 * np.sum(eps_o0**2) + 0.5 * self.alpha * np.sum(eps_x0**2)
        
        eps_o1 = self.observations[1] - self.W1 @ self.x1
        eps_x1 = self.x1
        F1 = 0.5 * np.sum(eps_o1**2) + 0.5 * self.alpha * np.sum(eps_x1**2)
        
        eps_o2 = self.observations[2] - self.W2 @ self.x2
        eps_x2 = self.x2
        F2 = 0.5 * np.sum(eps_o2**2) + 0.5 * self.alpha * np.sum(eps_x2**2)
        
        return float(F0 + F1 + F2)

    def step_predictive_coding(self, obs: List[np.ndarray]) -> None:
        """
        Performs one perception update step to reduce total free energy.
        """
        if not isinstance(obs, (list, tuple)) or len(obs) != 3:
            raise ValueError("Observations must be a sequence of 3 numpy arrays.")
            
        for idx, o in enumerate(obs):
            if not isinstance(o, np.ndarray) or o.shape != (5,):
                raise ValueError(f"Observation slice {idx} must be a numpy array of shape (5,).")
            if not np.isfinite(o).all():
                raise ValueError(f"Observation slice {idx} contains non-finite/NaN values.")

        self.observations = obs
        
        # x_i <- x_i + gamma * ( W_i.T @ (o_i - W_i @ x_i) - alpha * x_i )
        self.x0 = self.x0 + self.gamma * (self.W0.T @ (self.observations[0] - self.W0 @ self.x0) - self.alpha * self.x0)
        self.x1 = self.x1 + self.gamma * (self.W1.T @ (self.observations[1] - self.W1 @ self.x1) - self.alpha * self.x1)
        self.x2 = self.x2 + self.gamma * (self.W2.T @ (self.observations[2] - self.W2 @ self.x2) - self.alpha * self.x2)


def adjust_actuators(observations: List[np.ndarray]) -> Dict[str, float]:
    """
    Computes joint adjustments based on active predictive-coding error slices.
    """
    if len(observations) != 3:
        raise ValueError("Must provide exactly 3 observation slices.")
        
    o_flat = np.concatenate(observations)
    
    # Generate adjustments for joint_0, joint_1, joint_2
    adjustments = {
        f"joint_{j}": float(-0.5 * o_flat[j % len(o_flat)]) for j in range(3)
    }
    return adjustments
