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
