import numpy as np

class System1Reflex:
    """Low-latency reactive controller (Cerebellum) bypassing System 2 settling."""
    def __init__(self, collision_threshold=0.20, tilt_threshold=0.5):
        self.collision_threshold = collision_threshold
        self.tilt_threshold = tilt_threshold
        self.last_escape_time = 0.0
        
    def evaluate(self, sensory_state):
        if isinstance(sensory_state, dict):
            dist = sensory_state.get("dist", 0.0)
            tilt = sensory_state.get("tilt", 0.0)
            error_energy = sensory_state.get("error_energy", 0.0)
        elif hasattr(sensory_state, "dist") or hasattr(sensory_state, "tilt") or hasattr(sensory_state, "error_energy"):
            dist = getattr(sensory_state, "dist", 0.0)
            tilt = getattr(sensory_state, "tilt", 0.0)
            error_energy = getattr(sensory_state, "error_energy", 0.0)
        else:
            dist = sensory_state[0]
            tilt = sensory_state[1]
            error_energy = sensory_state[2]
        
        is_collision_hazard = dist < self.collision_threshold
        is_imbalance_hazard = abs(tilt) > self.tilt_threshold
        is_surprise_hazard = error_energy > 5.0
        
        if is_collision_hazard or is_imbalance_hazard or is_surprise_hazard:
            if is_collision_hazard:
                return True, np.array([0.0, -1.5])  # BACKWARD maneuver
            if is_imbalance_hazard:
                return True, np.array([-1.0, -1.0])  # STABILIZE maneuver
            return True, np.array([0.0, 0.0])  # RE-SETTLE standby
        return False, None
