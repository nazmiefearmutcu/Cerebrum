"""Smoke test for the NON-METRIC relational few-shot benchmark (spec failure-mode FM7).

This task is DESIGNED to break CEREBRUM's metric grid prior: the relational graph is a
random DIRECTED graph with asymmetric, non-commuting relation composition, so the grid's
2D path-integration (pos += relation_vector, a commutative metric algebra) cannot assign a
consistent code to a node reached via different relation-paths. We therefore DO NOT assert
that CEREBRUM wins — only that the generator and every runner emit valid, finite numbers.

This file includes smoke tests for both the standard RelationalGraph and TreeRelationalGraph.
"""
import numpy as np
from cerebrum.config import CerebrumConfig
from cerebrum.core_net import CerebrumCore
from benchmarks.tasks.relational import (
    RelationalGraph, TreeRelationalGraph, make_episode, run_cerebrum_episode,
)
from benchmarks.baselines.flat_prior_relational import run_flat_relational_episode
from benchmarks.baselines.backprop_mlp_relational import run_mlp_relational_episode


# =====================================================================
# Tests for standard RelationalGraph
# =====================================================================

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


def test_cerebrum_runner_finite_and_in_range():
    ep = make_episode(n_nodes=14, n_relations=3, vocab=5, K=12, seed=3)
    cfg = CerebrumConfig(dims=(5, 8, 8), grid_n_modules=8, n_settle=10, seed=0)
    s = run_cerebrum_episode(CerebrumCore(cfg), ep)
    assert np.isfinite(s) and 0.0 <= s <= 1.0


def test_flat_runner_finite_and_in_range():
    ep = make_episode(n_nodes=14, n_relations=3, vocab=5, K=12, seed=3)
    s = run_flat_relational_episode(ep)
    assert np.isfinite(s) and 0.0 <= s <= 1.0


def test_mlp_runner_finite_and_in_range():
    ep = make_episode(n_nodes=14, n_relations=3, vocab=5, K=12, seed=3)
    s = run_mlp_relational_episode(ep, epochs=50)
    assert np.isfinite(s) and 0.0 <= s <= 1.0


# =====================================================================
# Tests for TreeRelationalGraph
# =====================================================================

def test_tree_graph_is_directed_and_nonmetric():
    g = TreeRelationalGraph(n_nodes=12, n_relations=3, vocab=5, seed=0)
    
    # 1. Asymmetry: a --r--> b does not imply b --r--> a
    # Let's test parent and child transitions specifically
    # For node 0 (root), child 1 (r=1) is node 1:
    assert g.step(0, 1) == 1
    assert g.step(1, 1) == 3  # child 1 of node 1 is node 3, not node 0
    
    found_asym = False
    for a in range(g.n_nodes):
        for r in range(g.n_relations):
            b = g.step(a, r)
            if g.step(b, r) != a:
                found_asym = True
    assert found_asym, "tree graph must be asymmetric"
    
    # 2. Non-commuting: child relations 1 and 2 do not commute
    # Starting at root 0, Child 1 then Child 2 vs Child 2 then Child 1
    # step(step(0, 1), 2) = step(1, 2) = 4
    # step(step(0, 2), 1) = step(2, 1) = 5
    assert g.step(g.step(0, 1), 2) != g.step(g.step(0, 2), 1)
    
    # Obs deterministic per node, correct width
    assert np.array_equal(g.obs_at(3), g.obs_at(3))
    assert g.obs_at(0).shape == (5,)


def test_tree_relation_action_is_exogenous_label():
    g = TreeRelationalGraph(n_nodes=10, n_relations=3, vocab=5, seed=1)
    v0 = g.relation_vec(0)
    v0b = g.relation_vec(0)
    assert np.array_equal(v0, v0b)
    assert v0.shape == (2,)


def test_tree_episode_has_walk_and_heldout_queries():
    ep = make_episode(n_nodes=14, n_relations=3, vocab=5, K=12, seed=2, graph_class=TreeRelationalGraph)
    assert len(ep.walk) == 12
    # For small trees, sometimes we might not see enough nodes for queries. But with K=12,
    # seed=2, there should be some 2-hop compositions. Let's make sure we have queries or
    # at least check that the query components are correctly structured.
    assert len(ep.queries) > 0
    for (start, rel_path, target) in ep.queries:
        assert target in ep.observed_nodes
        assert len(rel_path) == 2  # all queries in make_episode are 2-hop compositions


def test_tree_cerebrum_runner_finite_and_in_range():
    ep = make_episode(n_nodes=14, n_relations=3, vocab=5, K=12, seed=3, graph_class=TreeRelationalGraph)
    cfg = CerebrumConfig(dims=(5, 8, 8), grid_n_modules=8, n_settle=10, seed=0)
    s = run_cerebrum_episode(CerebrumCore(cfg), ep)
    assert np.isfinite(s) and 0.0 <= s <= 1.0


def test_tree_flat_runner_finite_and_in_range():
    ep = make_episode(n_nodes=14, n_relations=3, vocab=5, K=12, seed=3, graph_class=TreeRelationalGraph)
    s = run_flat_relational_episode(ep)
    assert np.isfinite(s) and 0.0 <= s <= 1.0


def test_tree_mlp_runner_finite_and_in_range():
    ep = make_episode(n_nodes=14, n_relations=3, vocab=5, K=12, seed=3, graph_class=TreeRelationalGraph)
    s = run_mlp_relational_episode(ep, epochs=50)
    assert np.isfinite(s) and 0.0 <= s <= 1.0
