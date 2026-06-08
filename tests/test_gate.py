import numpy as np
from grail.config import GRAILConfig
from grail.gate import BasalGangliaGate
from grail.rng import SeededRNG
from grail.invariants import assert_one_hot

def test_bid_is_scalar_own_error_plus_excitability():
    g = BasalGangliaGate(n_modules=3, k_slots=2, cfg=GRAILConfig(), seed=0)
    bids = g.bid(err_sq=np.array([1.0,4.0,0.0]), pi=np.array([1.0,1.0,1.0]))
    assert bids.shape == (3,)
    assert bids[1] > bids[0] > bids[2]                  # higher own-error -> higher bid (no cross-module term)

def test_selection_is_one_hot_per_slot():
    g = BasalGangliaGate(n_modules=4, k_slots=2, cfg=GRAILConfig(), seed=1)
    z = g.select(bids=np.array([1.,2.,0.5,0.1]), rng=SeededRNG(0), T_gate=0.5)
    assert z.shape == (4,2); assert_one_hot(z, axis=0)

def test_selection_is_stochastic_not_argmax():
    g = BasalGangliaGate(n_modules=3, k_slots=1, cfg=GRAILConfig(), seed=2)
    wins = [int(np.argmax(g.select(np.array([1.0,0.9,0.8]), SeededRNG(s), T_gate=1.0)[:,0])) for s in range(50)]
    assert len(set(wins)) > 1                            # noise -> not always the top bidder (Pillar 4)
