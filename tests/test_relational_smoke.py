"""Smoke test for the NON-METRIC relational few-shot benchmark (spec failure-mode FM7).

This task is DESIGNED to break GRAIL's metric grid prior: the relational graph is a
random DIRECTED graph with asymmetric, non-commuting relation composition, so the grid's
2D path-integration (pos += relation_vector, a commutative metric algebra) cannot assign a
consistent code to a node reached via different relation-paths. We therefore DO NOT assert
that GRAIL wins — only that the generator and every runner emit valid, finite numbers.
"""
import numpy as np
from grail.config import GRAILConfig
from grail.network import GRAILCore
from benchmarks.tasks.relational import (
    RelationalGraph, make_episode, run_grail_episode,
    HierarchyGraph, make_hierarchy_episode,
)
from benchmarks.baselines.flat_prior_relational import run_flat_relational_episode
from benchmarks.baselines.backprop_mlp_relational import run_mlp_relational_episode


def test_graph_is_directed_and_nonmetric():
    g = RelationalGraph(n_nodes=12, n_relations=3, vocab=5, seed=0)
    # asymmetry: there exists a relation r and node a with a--r-->b but NOT b--r-->a
    found_asym = False
    for a in range(g.n_nodes):
        for r in range(g.n_relations):
            b = g.step(a, r)
            if g.step(b, r) != a:
                found_asym = True
    assert found_asym, "graph must be asymmetric (a--r-->b should not imply b--r-->a)"
    # obs deterministic per node, correct width
    assert np.array_equal(g.obs_at(3), g.obs_at(3))
    assert g.obs_at(0).shape == (5,)


def test_relation_action_is_exogenous_label():
    g = RelationalGraph(n_nodes=10, n_relations=3, vocab=5, seed=1)
    # the relation vector is a fixed external label, NOT derived from any node/obs/state
    v0 = g.relation_vec(0)
    v0b = g.relation_vec(0)
    assert np.array_equal(v0, v0b)            # deterministic, frozen per relation id
    assert v0.shape == (2,)


def test_episode_has_walk_and_heldout_queries():
    ep = make_episode(n_nodes=14, n_relations=3, vocab=5, K=12, seed=2)
    assert len(ep.walk) == 12
    assert len(ep.queries) > 0
    for (start, rel_path, target) in ep.queries:
        assert target in ep.observed_nodes        # held-out target's obs is known
        assert len(rel_path) >= 1                  # composed relation path


def test_grail_runner_finite_and_in_range():
    ep = make_episode(n_nodes=14, n_relations=3, vocab=5, K=12, seed=3)
    cfg = GRAILConfig(dims=(5, 8, 8), grid_n_modules=8, n_settle=10, seed=0)
    s = run_grail_episode(GRAILCore(cfg), ep)
    assert np.isfinite(s) and 0.0 <= s <= 1.0


def test_flat_runner_finite_and_in_range():
    ep = make_episode(n_nodes=14, n_relations=3, vocab=5, K=12, seed=3)
    s = run_flat_relational_episode(ep)
    assert np.isfinite(s) and 0.0 <= s <= 1.0


def test_mlp_runner_finite_and_in_range():
    ep = make_episode(n_nodes=14, n_relations=3, vocab=5, K=12, seed=3)
    s = run_mlp_relational_episode(ep, epochs=50)
    assert np.isfinite(s) and 0.0 <= s <= 1.0


def test_hierarchy_graph_properties():
    # n_nodes must be large enough to have left/right children (e.g. 10)
    g = HierarchyGraph(n_nodes=10, vocab=5, seed=0)
    # Check parent relation (0), left child (1), right child (2)
    assert g.n_nodes == 10
    assert g.n_relations == 3
    
    # 1. Asymmetry: there exists r, a, b with a--r-->b but NOT b--r-->a
    found_asym = False
    for a in range(g.n_nodes):
        for r in range(g.n_relations):
            b = g.step(a, r)
            if g.step(b, r) != a:
                found_asym = True
    assert found_asym, "HierarchyGraph must be asymmetric"

    # 2. Non-commuting transitions: step(step(a, r1), r2) != step(step(a, r2), r1)
    found_non_commute = False
    for a in range(g.n_nodes):
        for r1 in range(g.n_relations):
            for r2 in range(g.n_relations):
                # follow r1 then r2
                n12 = g.step(g.step(a, r1), r2)
                # follow r2 then r1
                n21 = g.step(g.step(a, r2), r1)
                if n12 != n21:
                    found_non_commute = True
    assert found_non_commute, "HierarchyGraph transitions must not compose as commuting grid rotations"

    # 3. Directed properties: some transitions are directed and do not trivially loop
    # Check that left child of 0 is 1, parent of 1 is 0
    assert g.step(0, 1) == 1
    assert g.step(1, 0) == 0
    # Check parent of 0 is 0
    assert g.step(0, 0) == 0
    # Check left child of 9 (since 2*9+1 = 19 >= 10, should return 9)
    assert g.step(9, 1) == 9

    # obs deterministic per node, correct width
    assert np.array_equal(g.obs_at(3), g.obs_at(3))
    assert g.obs_at(0).shape == (5,)


def test_hierarchy_runners():
    # run_grail_episode, run_flat_relational_episode, and run_mlp_relational_episode
    # all run without error on a hierarchy episode
    ep = make_hierarchy_episode(n_nodes=10, vocab=5, K=8, seed=4)
    
    # 1. GRAIL runner
    cfg = GRAILConfig(dims=(5, 8, 8), grid_n_modules=8, n_settle=10, seed=0)
    s_grail = run_grail_episode(GRAILCore(cfg), ep)
    assert np.isfinite(s_grail) and 0.0 <= s_grail <= 1.0
    
    # 2. Flat prior runner
    s_flat = run_flat_relational_episode(ep)
    assert np.isfinite(s_flat) and 0.0 <= s_flat <= 1.0
    
    # 3. MLP runner
    s_mlp = run_mlp_relational_episode(ep, epochs=10)
    assert np.isfinite(s_mlp) and 0.0 <= s_mlp <= 1.0
