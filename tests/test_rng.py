import numpy as np
from cerebrum.rng import SeededRNG

def test_reproducible():
    a = SeededRNG(123).normal((4,)); b = SeededRNG(123).normal((4,))
    assert np.allclose(a, b)

def test_zeroable_for_deterministic_tests():
    r = SeededRNG(1, enabled=False)
    assert np.allclose(r.normal((5,)), 0.0)   # disabling noise gives exact zeros (deterministic limit)
