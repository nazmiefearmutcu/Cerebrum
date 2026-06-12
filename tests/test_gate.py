import numpy as np
from cerebrum.config import CerebrumConfig
from cerebrum.gate import BasalGangliaGate
from cerebrum.rng import SeededRNG
from cerebrum.invariants import assert_one_hot

def test_bid_is_scalar_own_error_plus_excitability():
    g = BasalGangliaGate(n_modules=3, k_slots=2, cfg=CerebrumConfig(), seed=0)
    bids = g.bid(err_sq=np.array([1.0,4.0,0.0]), pi=np.array([1.0,1.0,1.0]))
    assert bids.shape == (3,)
    assert bids[1] > bids[0] > bids[2]                  # higher own-error -> higher bid (no cross-module term)

def test_selection_is_one_hot_per_slot():
    g = BasalGangliaGate(n_modules=4, k_slots=2, cfg=CerebrumConfig(), seed=1)
    z = g.select(bids=np.array([1.,2.,0.5,0.1]), rng=SeededRNG(0), T_gate=0.5)
    assert z.shape == (4,2); assert_one_hot(z, axis=0)

def test_selection_is_stochastic_not_argmax():
    g = BasalGangliaGate(n_modules=3, k_slots=1, cfg=CerebrumConfig(), seed=2)
    wins = [int(np.argmax(g.select(np.array([1.0,0.9,0.8]), SeededRNG(s), T_gate=1.0)[:,0])) for s in range(50)]
    assert len(set(wins)) > 1                            # noise -> not always the top bidder (Pillar 4)

def test_local_learning_raises_win_prob_of_rewarded_module():
    g = BasalGangliaGate(n_modules=3, k_slots=1, cfg=CerebrumConfig(eta_w=0.5), seed=3)
    rng = SeededRNG(0)
    # repeatedly: module 1 wins and is rewarded (M>0) -> its Go weight should grow
    G1_before = float(g.G[1,0].item())
    for _ in range(30):
        z = g.select(np.array([0.5,1.0,0.5]), rng, T_gate=0.5)
        reward = 1.0 if z[1, 0] > 0.5 else 0.0
        g.learn(M=reward)
    assert float(g.G[1,0].item()) > G1_before                          # rewarded winner's Go weight increases (scalar M)

def test_lam_g_decays_gate_weights_toward_init():
    g = BasalGangliaGate(n_modules=3, k_slots=1, cfg=CerebrumConfig(eta_w=0.0, lam_g=0.5), seed=0)
    g.N[:] = 1.0                                          # push NoGo away from its 0 init
    g.select(np.array([1.0, 1.0, 1.0]), SeededRNG(0), T_gate=0.5)
    g.learn(M=0.0)                                        # eta*M=0 -> only the lam_g decay acts
    assert np.all(g.N.cpu().numpy() < 1.0)                              # NoGo decays toward its 0 init
    assert np.all(np.abs(g.G.cpu().numpy() - 0.5) <= 0.1 + 1e-9)       # Go pulled within the decayed range of its 0.5 init

def test_reward_aware_homeostasis_spares_rewarded_wins():
    g = BasalGangliaGate(n_modules=2, k_slots=1, cfg=CerebrumConfig(), seed=0)
    g.select(np.array([2.0, 0.1]), SeededRNG(0), T_gate=0.2)
    g.homeostasis(M=3.0)                                  # rewarded win -> winner NOT penalized as a hog
    g2 = BasalGangliaGate(n_modules=2, k_slots=1, cfg=CerebrumConfig(), seed=0)
    g2.select(np.array([2.0, 0.1]), SeededRNG(0), T_gate=0.2)
    g2.homeostasis(M=None)                                # plain anti-hog penalizes the winner
    winner = int(np.argmax(g._z.cpu().numpy()[:, 0]))
    assert float(g.theta[winner].item()) > float(g2.theta[winner].item())             # reward-aware spares the correct winner

def test_homeostasis_raises_excitability_of_starved_module():
    g = BasalGangliaGate(n_modules=3, k_slots=1, cfg=CerebrumConfig(), seed=4)
    rng = SeededRNG(1)
    for _ in range(40):
        g.select(np.array([5.0, 0.0, 0.0]), rng, T_gate=0.2)  # module 0 hogs the slot
        g.homeostasis()
    assert float(g.theta[1].item()) > 0 and float(g.theta[2].item()) > 0              # starved modules' excitability rises (anti-dead-expert)
    assert float(g.theta[0].item()) < float(g.theta[1].item())                        # the hog's excitability is suppressed
