import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class Exogenous:
    """An action/motor signal that is, by construction, NOT a function of network state.
    Only values explicitly wrapped here (from the task/environment) can drive the grid
    transition. This makes a data-dependent z_act a type error (BAN-1)."""
    value: np.ndarray
    def __post_init__(self):
        v = np.asarray(self.value, dtype=float)
        object.__setattr__(self, "value", v)
