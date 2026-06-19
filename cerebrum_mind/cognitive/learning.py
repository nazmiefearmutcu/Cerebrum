import numpy as np

class HebbianMetaplasticity:
    """
    Implements Hebbian updates with metaplastic constraints (clamping).
    """
    def __init__(self, num_synapses: int):
        self.weights = np.full((num_synapses,), 0.5, dtype=float)

    def learn_step(self, post_activity: np.ndarray, pre_activity: np.ndarray, neuromodulator: float, eta: float = 0.1) -> None:
        """
        Updates weights based on pre/post activity and a surprise/neuromodulator signal:
        w_i <- w_i + eta * M * pre_i * post_i
        """
        if neuromodulator == 0.0:
            return

        # Update weights: M is neuromodulator, x is pre_activity, y is post_activity
        # Ensure array shapes align or element-wise matches
        self.weights = self.weights + eta * neuromodulator * pre_activity * post_activity
        
        # Clamp between 0.0 and 1.0 (metaplastic limits)
        self.weights = np.clip(self.weights, 0.0, 1.0)
