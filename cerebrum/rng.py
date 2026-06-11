import numpy as np

class SeededRNG:
    """Centralized reproducible noise. enabled=False -> exact zeros (deterministic limit for tests)."""
    def __init__(self, seed: int = 0, enabled: bool = True):
        self._rng = np.random.default_rng(seed)
        self.enabled = enabled
    def normal(self, shape, scale: float = 1.0):
        if not self.enabled:
            return np.zeros(shape)
        return self._rng.normal(0.0, scale, size=shape)
    def gumbel(self, shape):
        if not self.enabled:
            return np.zeros(shape)
        return self._rng.gumbel(0.0, 1.0, size=shape)
    def uniform(self, shape):
        return self._rng.uniform(0.0, 1.0, size=shape)
