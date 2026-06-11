import numpy as np
from cerebrum.config import CerebrumConfig
from cerebrum.neuromod import Neuromodulator
from cerebrum.invariants import assert_scalar_M

def test_M_is_reward_minus_baseline_and_scalar():
    nm = Neuromodulator(CerebrumConfig())
    M = nm.update(reward=1.0); assert_scalar_M(M)
    assert M > 0                       # first reward above baseline 0 -> positive surprise

def test_baseline_tracks_reward():
    nm = Neuromodulator(CerebrumConfig(tau_r=2.0))
    for _ in range(500): nm.update(reward=1.0)
    assert abs(nm.r_bar - 1.0) < 1e-2  # steady reward -> baseline converges -> M -> 0
    assert abs(nm.update(1.0)) < 1e-2

def test_couplings_monotone():
    nm = Neuromodulator(CerebrumConfig())
    assert nm.temperature(0.5) > nm.temperature(0.0)   # surprise heats up
    assert nm.eta(0.5) > nm.eta(0.0)                    # surprise raises learning rate
