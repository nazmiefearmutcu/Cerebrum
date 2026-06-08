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

def test_local_learning_raises_win_prob_of_rewarded_module():
    g = BasalGangliaGate(n_modules=3, k_slots=1, cfg=GRAILConfig(eta_w=0.5), seed=3)
    rng = SeededRNG(0)
    # repeatedly: module 1 wins and is rewarded (M>0) -> its Go weight should grow
    G1_before = g.G[1,0]
    for _ in range(30):
        g.select(np.array([0.5,1.0,0.5]), rng, T_gate=0.5)
        g.learn(M=1.0)
    assert g.G[1,0] > G1_before                          # rewarded winner's Go weight increases (scalar M)

def test_homeostasis_raises_excitability_of_starved_module():
    g = BasalGangliaGate(n_modules=3, k_slots=1, cfg=GRAILConfig(), seed=4)
    rng = SeededRNG(1)
    for _ in range(40):
        g.select(np.array([5.0, 0.0, 0.0]), rng, T_gate=0.2)  # module 0 hogs the slot
        g.homeostasis()
    assert g.theta[1] > 0 and g.theta[2] > 0              # starved modules' excitability rises (anti-dead-expert)
    assert g.theta[0] < g.theta[1]                        # the hog's excitability is suppressed
