import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks.tasks.relational import TreeRelationalGraph, make_episode

def test_properties(n_nodes, n_relations, seed):
    g = TreeRelationalGraph(n_nodes=n_nodes, n_relations=n_relations, vocab=10, seed=seed)
    
    # 1. Verify asymmetry:
    # A graph is asymmetric if there exists some transition a --r--> b such that b --r--> a is False (for a != b).
    # Since we have parent and child relations, we check if there is at least one node and one relation
    # where step(step(a, r), r) != a (excluding trivial self-loops where step(a,r) == a).
    asymmetric = False
    for r in range(n_relations):
        for u in range(n_nodes):
            v = g.step(u, r)
            if v != u: # exclude trivial self-loops
                if g.step(v, r) != u:
                    asymmetric = True
                    break
        if asymmetric:
            break
            
    # 2. Verify non-commutation:
    # A relation r is an identity map if for all u, step(u, r) == u.
    # If a relation is the identity map, it commutes with all other relations.
    # Therefore, a pair {r1, r2} can only be non-commuting if NEITHER r1 nor r2 is the identity map.
    # We call such pairs "active pairs".
    identity_relations = set()
    for r in range(n_relations):
        is_identity = True
        for u in range(n_nodes):
            if g.step(u, r) != u:
                is_identity = False
                break
        if is_identity:
            identity_relations.add(r)

    active_pairs = 0
    non_commuting_pairs = 0
    total_pairs = 0
    for r1 in range(n_relations):
        for r2 in range(r1 + 1, n_relations):
            total_pairs += 1
            if r1 not in identity_relations and r2 not in identity_relations:
                active_pairs += 1
                
            commutes = True
            for u in range(n_nodes):
                dest1 = g.step(g.step(u, r1), r2)
                dest2 = g.step(g.step(u, r2), r1)
                if dest1 != dest2:
                    commutes = False
                    break
            if not commutes:
                non_commuting_pairs += 1

    # 3. Verify non-metric (path-dependence / path-collapse fails):
    # If step(step(u, r1), r2) != step(step(u, r2), r1), then the same relation vector sum leads to
    # two different nodes. This is a direct proof that the space is mathematically non-metric.
    non_metric_triplets = 0
    for r1 in range(n_relations):
        for r2 in range(r1 + 1, n_relations):
            for u in range(n_nodes):
                dest1 = g.step(g.step(u, r1), r2)
                dest2 = g.step(g.step(u, r2), r1)
                if dest1 != dest2:
                    non_metric_triplets += 1

    return {
        "asymmetric": asymmetric,
        "non_commuting_pairs": non_commuting_pairs,
        "active_pairs": active_pairs,
        "total_pairs": total_pairs,
        "non_metric_triplets": non_metric_triplets,
        "identity_relations": sorted(list(identity_relations))
    }

def run_verification():
    seeds = [0, 1, 42, 100, 999]
    sizes = [5, 10, 20, 50, 100]
    relation_counts = [2, 3, 5, 10]
    
    print("Starting verification of TreeRelationalGraph properties...")
    all_passed = True
    
    for seed in seeds:
        for size in sizes:
            for n_rel in relation_counts:
                res = test_properties(size, n_rel, seed)
                
                asym_ok = res["asymmetric"] or size <= 1
                # Non-commuting: all active pairs must be non-commuting (when active_pairs > 0)
                comm_ok = (res["non_commuting_pairs"] == res["active_pairs"]) or (size <= 2)
                # Non-metric: we must have non-metric triplets if active_pairs > 0 and size > 2
                metric_ok = (res["non_metric_triplets"] > 0) or (res["active_pairs"] == 0) or (size <= 2)
                
                if not (asym_ok and comm_ok and metric_ok):
                    print(f"FAIL: Seed={seed}, Size={size}, Relations={n_rel} -> {res}")
                    all_passed = False
                else:
                    print(f"PASS: Seed={seed:3d}, Size={size:3d}, Relations={n_rel:2d} | Asym: {res['asymmetric']} | Non-commuting/Active pairs: {res['non_commuting_pairs']}/{res['active_pairs']} (total: {res['total_pairs']}) | Triplets: {res['non_metric_triplets']} | Identity rels: {res['identity_relations']}")
                    
    # Edge case: n_relations < 2 must raise AssertionError
    try:
        TreeRelationalGraph(n_nodes=10, n_relations=1, vocab=5, seed=0)
        print("FAIL: n_relations < 2 did not raise AssertionError")
        all_passed = False
    except AssertionError as e:
        print(f"PASS: n_relations < 2 correctly raised AssertionError: {e}")
        
    try:
        TreeRelationalGraph(n_nodes=10, n_relations=0, vocab=5, seed=0)
        print("FAIL: n_relations = 0 did not raise AssertionError")
        all_passed = False
    except AssertionError as e:
        print(f"PASS: n_relations = 0 correctly raised AssertionError: {e}")

    if all_passed:
        print("\nVerification SUCCESS: All configurations satisfy non-metric, asymmetric, and non-commuting properties.")
    else:
        print("\nVerification FAILURE: Some configurations did not satisfy the properties.")

if __name__ == "__main__":
    run_verification()
