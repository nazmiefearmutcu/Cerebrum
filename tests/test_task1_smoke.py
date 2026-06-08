import numpy as np
from benchmarks.tasks.gridworld import GridWorld, make_episode

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
