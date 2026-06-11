import numpy as np

class MotorProcessor:
    """Maps continuous workspace actions or routing indexes to wheel velocities."""
    def __init__(self, mode="discrete", W_motor=None, b_motor=None, u_sat=2.0):
        self.mode = mode
        self.W_motor = W_motor
        self.b_motor = b_motor
        self.u_sat = u_sat
        
    def process(self, action_vector):
        if action_vector is None:
            action_vector = np.array([])
        else:
            action_vector = np.asarray(action_vector, dtype=float)
            action_vector = np.where(np.isnan(action_vector) | np.isinf(action_vector), 0.0, action_vector)
            
        if len(action_vector) == 0:
            return np.array([0.0, 0.0])
            
        if self.mode == "linear" and self.W_motor is not None:
            try:
                # Linear readout mapping
                W_motor = np.asarray(self.W_motor)
                if W_motor.ndim >= 3:
                    W_motor = W_motor.reshape(W_motor.shape[0], -1)
                
                if W_motor.ndim == 0:
                    vels = np.zeros(2)
                elif W_motor.ndim == 1:
                    expected_dim = W_motor.shape[0]
                    if len(action_vector) != expected_dim:
                        vels = np.zeros(2)
                    else:
                        res = np.dot(W_motor, action_vector)
                        vels = np.array([res, 0.0]) if np.isscalar(res) else np.asarray(res)
                else:
                    expected_dim = W_motor.shape[1]
                    if len(action_vector) != expected_dim:
                        output_dim = W_motor.shape[0]
                        vels = np.zeros(output_dim)
                    else:
                        b_val = self.b_motor if self.b_motor is not None else np.zeros(W_motor.shape[0])
                        b_val = np.asarray(b_val, dtype=float)
                        if b_val.ndim >= 2:
                            b_val = b_val.flatten()
                        if b_val.shape[0] != W_motor.shape[0]:
                            b_val = np.zeros(W_motor.shape[0])
                        vels = np.dot(W_motor, action_vector) + b_val
            except ValueError:
                vels = np.zeros(2)
        else:
            # Discrete workspace gating mapping (Default Mock)
            if np.all(action_vector == 0.0):
                vels = np.array([0.0, 0.0])  # Standby
            else:
                act_idx = np.argmax(action_vector)
                if act_idx == 0:
                    vels = np.array([1.0, 1.0])  # Forward
                elif act_idx == 1:
                    vels = np.array([-0.5, 0.5])  # Left turn
                elif act_idx == 2:
                    vels = np.array([0.5, -0.5])  # Right turn
                else:
                    vels = np.array([0.0, 0.0])  # Standby
                
        return np.clip(vels, -self.u_sat, self.u_sat)
