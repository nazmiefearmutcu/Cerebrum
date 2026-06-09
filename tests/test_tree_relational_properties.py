import pytest
import numpy as np
from benchmarks.tasks.relational import TreeRelationalGraph

def test_tree_graph_asymmetry_extensive():
    # Verify asymmetry across multiple seeds, tree sizes, and relation counts
    seeds = [0, 1, 42, 123]
    tree_sizes = [5, 10, 30, 100]
    relation_counts = [2, 3, 5, 10]

    for seed in seeds:
        for n_nodes in tree_sizes:
            for n_relations in relation_counts:
                g = TreeRelationalGraph(n_nodes=n_nodes, n_relations=n_relations, vocab=5, seed=seed)
                # For any node u and relation r, if it transitions to a different node v,
                # then v transitioning via the SAME relation r must not lead back to u.
                for u in range(g.n_nodes):
                    for r in range(g.n_relations):
                        v = g.step(u, r)
                        if v != u:
                            assert g.step(v, r) != u, (
                                f"Symmetry found! u={u} --r={r}--> v={v} --r={r}--> u={g.step(v, r)} "
                                f"for seed={seed}, n_nodes={n_nodes}, n_relations={n_relations}"
                            )

def test_tree_graph_non_commuting_extensive():
    # Verify non-commuting property across multiple seeds, tree sizes, and relation counts
    seeds = [0, 1, 42, 123]
    tree_sizes = [5, 10, 30, 100]
    # Relation counts >= 2 (for parent and at least one child)
    relation_counts = [2, 3, 5, 10]

    for seed in seeds:
        for n_nodes in tree_sizes:
            for n_relations in relation_counts:
                g = TreeRelationalGraph(n_nodes=n_nodes, n_relations=n_relations, vocab=5, seed=seed)
                B = n_relations - 1
                
                # 1. Non-commuting: Parent (0) and child (r >= 1) relations
                # We expect that for some node u, step(step(u, 0), r) != step(step(u, r), 0)
                # Let's verify that a non-commuting pair exists.
                found_non_comm_parent_child = False
                for u in range(g.n_nodes):
                    for r in range(1, n_relations):
                        # Apply parent then child r
                        v1 = g.step(g.step(u, 0), r)
                        # Apply child r then parent
                        v2 = g.step(g.step(u, r), 0)
                        if v1 != v2:
                            found_non_comm_parent_child = True
                            break
                    if found_non_comm_parent_child:
                        break
                # A tree with at least 2 nodes and n_relations >= 2 must have at least one non-commuting parent-child instance
                if n_nodes > 1:
                    assert found_non_comm_parent_child, (
                        f"All parent-child relations commuted for seed={seed}, n_nodes={n_nodes}, n_relations={n_relations}"
                    )

                # 2. Non-commuting: Two different child relations r1, r2 (only if n_relations >= 3, i.e. B >= 2)
                if n_relations >= 3:
                    found_non_comm_children = False
                    for u in range(g.n_nodes):
                        for r1 in range(1, n_relations):
                            for r2 in range(1, n_relations):
                                if r1 == r2:
                                    continue
                                v1 = g.step(g.step(u, r1), r2)
                                v2 = g.step(g.step(u, r2), r1)
                                if v1 != v2:
                                    found_non_comm_children = True
                                    break
                            if found_non_comm_children:
                                break
                        if found_non_comm_children:
                            break
                    if n_nodes > B:
                        assert found_non_comm_children, (
                            f"All child-child relations commuted for seed={seed}, n_nodes={n_nodes}, n_relations={n_relations}"
                        )

def test_tree_graph_non_metric_vectors():
    # Verify that the vector representation violates metric/Euclidean properties.
    # Specifically, a path cycle u -> v -> u has a non-zero cumulative relation vector sum.
    seeds = [0, 1, 42, 123]
    for seed in seeds:
        g = TreeRelationalGraph(n_nodes=10, n_relations=3, vocab=5, seed=seed)
        # Cycle: 0 -> 1 (child 1) -> 0 (parent)
        # The path is relation 1 then relation 0.
        # The sum of relation vectors is g.relation_vec(1) + g.relation_vec(0)
        v_sum = g.relation_vec(1) + g.relation_vec(0)
        # It must not be zero vector (Euclidean metric displacement for a cycle must be zero)
        assert np.linalg.norm(v_sum) > 1e-5, f"Vector sum was zero for seed={seed}"

def test_tree_graph_edge_cases():
    # Case 1: n_relations < 2 raises AssertionError
    with pytest.raises(AssertionError):
        TreeRelationalGraph(n_nodes=10, n_relations=1, vocab=5)
    with pytest.raises(AssertionError):
        TreeRelationalGraph(n_nodes=10, n_relations=0, vocab=5)

    # Case 2: Leaf nodes trapping
    # Let's inspect a tree with n_nodes=5, n_relations=3 (B=2)
    # Nodes:
    # 0: parent 0, children: 1, 2
    # 1: parent 0, children: 3, 4
    # 2: parent 0, children: 5, 6 -> both out of bounds, so they must self-loop (trap)
    # 3: parent 1, children: 7, 8 -> trap
    # 4: parent 1, children: 9, 10 -> trap
    g = TreeRelationalGraph(n_nodes=5, n_relations=3, vocab=5, seed=0)
    
    # Node 2 is a leaf because its child indices would be 2*2 + 1 = 5 and 2*2 + 2 = 6, both >= 5
    assert g.step(2, 1) == 2, "Leaf node 2 child 1 did not self-loop"
    assert g.step(2, 2) == 2, "Leaf node 2 child 2 did not self-loop"
    # But parent relation escapes!
    assert g.step(2, 0) == 0, "Leaf node 2 parent did not escape to 0"

    # Node 3 is a leaf
    assert g.step(3, 1) == 3
    assert g.step(3, 2) == 3
    assert g.step(3, 0) == 1, "Leaf node 3 parent did not escape to 1"

    # Case 3: Root node behavior
    assert g.step(0, 0) == 0, "Root node parent did not self-loop"

    # Case 4: n_nodes = 1 (minimal tree)
    g_mini = TreeRelationalGraph(n_nodes=1, n_relations=2, vocab=5, seed=0)
    assert g_mini.step(0, 0) == 0
    assert g_mini.step(0, 1) == 0
    assert np.array_equal(g_mini.obs_at(0), g_mini.obs_at(0))
