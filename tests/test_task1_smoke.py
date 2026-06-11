import numpy as np
from benchmarks.tasks.gridworld import GridWorld, make_episode
from cerebrum.config import CerebrumConfig
from cerebrum.network import CerebrumCore
from benchmarks.tasks.graph_completion import run_cerebrum_episode

def test_gridworld_obs_are_consistent_per_cell():
    gw = GridWorld(h=4, w=4, vocab=5, seed=0)
    assert np.array_equal(gw.obs_at((1,2)), gw.obs_at((1,2)))   # deterministic per cell
    assert gw.obs_at((0,0)).shape == (5,)

def test_episode_has_walk_and_heldout_queries():
    ep = make_episode(h=4, w=4, vocab=5, K=8, seed=1)
    assert len(ep.walk) == 8
    assert len(ep.queries) > 0
    for (start, disp, target_cell) in ep.queries:
        assert target_cell in ep.observed_cells          # target was observed (obs known)

def test_cerebrum_scores_above_chance_on_completion():
    ep = make_episode(h=4, w=4, vocab=5, K=12, seed=2)
    cfg = CerebrumConfig(dims=(5,8,8), grid_n_modules=8, n_settle=10, seed=0)
    score = run_cerebrum_episode(CerebrumCore(cfg), ep)
    assert score > 1.0/5                       # beats 1/vocab chance via path-integrated completion

def test_flat_prior_is_near_chance_on_completion():
    from benchmarks.baselines.flat_prior import run_flat_episode
    from benchmarks.tasks.gridworld import make_episode
    ep = make_episode(h=4, w=4, vocab=5, K=12, seed=2)
    score = run_flat_episode(ep)
    assert score <= 0.45        # no path-integration -> cannot complete held-out paths

def test_backprop_mlp_runs():
    from benchmarks.baselines.backprop_mlp import run_mlp_episode
    from benchmarks.tasks.gridworld import make_episode
    ep = make_episode(h=4, w=4, vocab=5, K=12, seed=2)
    s = run_mlp_episode(ep, epochs=50)
    assert 0.0 <= s <= 1.0

def test_cerebrum_grid_beats_flat_prior_averaged():
    from benchmarks.run_task1 import run_sweep
    res = run_sweep(Ks=(5,10,20), seeds=(0,1,2), h=4, w=4, vocab=5)
    for K in (5,10,20):
        assert res["cerebrum"][K] > res["flat"][K] + 0.05    # grid prior buys sample efficiency
