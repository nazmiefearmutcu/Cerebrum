"""TRANSITIVE INFERENCE benchmark — a metric/linear-order few-shot task.

A total order of N items A>B>C>...  is laid out on a 1D LINE: item with rank r
sits at line coordinate r (rank 0 is the "greatest"/top of the order, larger rank
is "lesser"). Each item carries a fixed random sensory observation (a one-hot over
a sensory vocabulary, so the obs itself carries NO ordinal information — the order
lives only in WHERE on the line the item was encountered).

TRAINING shows only ADJACENT pairs (A>B, B>C, C>D, ...), each a few times. The
walk along the order is encoded for the grid HEAD as EXOGENOUS +1 / -1 actions on
the line axis (this is the grid's native metric/linear structure): we step from
rank r to rank r+1 by `Exogenous([+1, 0])`, binding the item's obs at the canonical
line coordinate of its rank. The grid therefore places items on a line by
path-integration, exactly the structure a transitive (linear) order needs.

TEST queries NON-ADJACENT pairs the model never saw together (e.g. B vs D): given
two items' observations, decide which is GREATER (lower rank). "Knowing the order"
== being able to place each probe item back on the line and compare positions.

The accuracy metric is the fraction of held-out NON-ADJACENT comparisons answered
correctly (which of the two probed items has the lower rank). Chance = 0.5.

BAN compliance: the line "step" fed to the grid is always an Exogenous(...) external
action (the rank index, supplied by the task), never derived from network state or
from the observations.
"""
import numpy as np
from dataclasses import dataclass


class LinearOrder:
    """A total order of N items on a 1D line. Item at rank r (0..N-1) has a fixed
    random one-hot sensory observation over `vocab` symbols. Rank 0 = greatest."""

    def __init__(self, n_items, vocab, seed=0):
        assert vocab >= n_items, "need a distinct sensory symbol per item"
        self.n = n_items
        self.vocab = vocab
        rng = np.random.default_rng(seed)
        # assign each rank a distinct sensory symbol (random permutation -> obs carries no order)
        self.symbol = rng.permutation(vocab)[:n_items]   # symbol[r] = sensory id of rank r

    def obs_at_rank(self, r):
        v = np.zeros(self.vocab)
        v[self.symbol[r]] = 1.0
        return v

    def rank_of_symbol(self, sym):
        return int(np.where(self.symbol == sym)[0][0])


@dataclass
class TransitiveEpisode:
    order: LinearOrder
    walk: list          # list of (rank, step_vec) adjacent training steps along the line
    exposures: int      # number of times each adjacent pair was shown
    train_pairs: set    # set of frozenset({rank_i, rank_j}) shown adjacently in training
    queries: list       # list of (rank_a, rank_b) NON-adjacent held-out comparisons


def make_episode(n_items=7, vocab=10, exposures=2, seed=0):
    """Build a transitive-inference episode.

    walk: traverse 0->1->...->(N-1) `exposures` times, each step an exogenous +1 on
    the line; binding happens at each rank's canonical coordinate. Only adjacent
    transitions appear (that is the few-shot supervision).

    queries: all NON-adjacent pairs (|rank_i - rank_j| >= 2) — never seen together.
    """
    order = LinearOrder(n_items, vocab, seed=seed)
    walk = []
    train_pairs = set()
    for _ in range(exposures):
        for r in range(n_items):
            # at rank r we are at line coordinate r; the step that BROUGHT us here is +1
            walk.append((r, np.array([1.0, 0.0])))
            if r + 1 < n_items:
                train_pairs.add(frozenset({r, r + 1}))
    queries = []
    for a in range(n_items):
        for b in range(a + 1, n_items):
            if b - a >= 2:                      # non-adjacent only
                queries.append((a, b))
    return TransitiveEpisode(order=order, walk=walk, exposures=exposures,
                             train_pairs=train_pairs, queries=queries)
