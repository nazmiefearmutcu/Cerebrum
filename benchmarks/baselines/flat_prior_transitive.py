"""Flat-prior baseline for TRANSITIVE INFERENCE.

A non-metric associative memory: it stores the directed adjacent comparisons it was
SHOWN (rank r is greater than rank r+1) as a key->value table over item symbols, with
NO line / no path-integration. Asked about a non-adjacent pair it never saw together,
it has no transition algebra to chain r>r+1>r+2, so it can only fall back to a
deterministic guess. This is the memorization comparator: it should sit at chance on
held-out non-adjacent comparisons.
"""
import numpy as np


def run_flat_episode(ep):
    order = ep.order
    # store directed "greater-than" facts only for SHOWN adjacent pairs (symbol-keyed).
    greater = set()    # set of (sym_greater, sym_lesser)
    for pair in ep.train_pairs:
        lo, hi = sorted(pair)               # lo rank < hi rank -> lo is greater
        greater.add((order.symbol[lo], order.symbol[hi]))
    correct = 0
    for (a, b) in ep.queries:
        sa, sb = order.symbol[a], order.symbol[b]
        if (sa, sb) in greater:
            pred_a_greater = True
        elif (sb, sa) in greater:
            pred_a_greater = False
        else:
            # never co-observed and no chaining -> deterministic fallback (coin in expectation)
            pred_a_greater = sa < sb
        truth_a_greater = (a < b)
        if pred_a_greater == truth_a_greater:
            correct += 1
    return correct / len(ep.queries) if ep.queries else 0.0
