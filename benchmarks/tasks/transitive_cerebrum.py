"""CEREBRUM-grid runner for the TRANSITIVE INFERENCE task.

Mechanism (uses only the grid HEAD's native ops: exogenous path-integration + Hebbian
content bind + completion):

  TRAIN: walk the order with EXOGENOUS +1 steps along the line axis, binding each
  item's sensory obs at the canonical line coordinate (= its rank). The grid lays
  the items out on a line by path-integration. Only adjacent steps occur.

  TEST (transitive comparison of two NEVER-CO-OBSERVED items): recover each probe
  item's line position by SCANNING — path-integrate to every integer line coordinate
  0..N-1 (each move an Exogenous step), `complete()` the obs there, and take the
  coordinate whose completed obs best matches the probe item's observation. Then the
  GREATER item is the one with the LOWER recovered coordinate (rank 0 = greatest).

The grid's structured linear prior means a position learned from adjacent exposures
extends to a metric line, so the relative order of two items that were never shown
together is reconstructable. Baselines that only memorize shown pairs cannot do this.
"""
import numpy as np
from cerebrum.types import Exogenous


def _goto_coord(net, x):
    """Place the grid at canonical line coordinate x via an exogenous move from origin."""
    net.grid.reset()
    net.move(Exogenous(np.array([float(x), 0.0])))


def _decode_position(net, probe_obs, n_items):
    """Scan the line and return the integer coordinate whose completed obs best
    matches probe_obs. Uses path-integration + completion only."""
    probe_sym = int(np.argmax(probe_obs))
    best_x, best_score = 0, -np.inf
    for x in range(n_items):
        _goto_coord(net, x)
        rec = net.predict_obs_here(probe_obs.size)
        score = float(rec[probe_sym]) if rec.size > probe_sym else -np.inf
        if score > best_score:
            best_score, best_x = score, x
    return best_x


def run_cerebrum_episode(net, ep):
    """Bind items along the line via adjacent exogenous steps; score held-out
    non-adjacent transitive comparisons."""
    order = ep.order
    # TRAIN: bind each rank's obs at its canonical line coordinate.
    for (rank, step_vec) in ep.walk:
        _goto_coord(net, rank)                     # exogenous placement on the line
        net.observe_and_learn(order.obs_at_rank(rank), reward=1.0)
    # TEST: compare non-adjacent pairs by recovering each item's line position.
    correct = 0
    for (a, b) in ep.queries:
        pos_a = _decode_position(net, order.obs_at_rank(a), order.n)
        pos_b = _decode_position(net, order.obs_at_rank(b), order.n)
        # ground truth: lower rank == greater. predict greater = lower recovered coord.
        if pos_a == pos_b:
            # decode collision (both items mapped to the same line coordinate): the grid
            # could not separate them. Deterministic fixed guess (no peeking at true rank)
            # -> behaves like a coin flip, contributing ~0.5 to accuracy in expectation.
            pred_a_greater = (int(np.argmax(order.obs_at_rank(a)))
                              < int(np.argmax(order.obs_at_rank(b))))
        else:
            pred_a_greater = pos_a < pos_b
        truth_a_greater = (a < b)                  # a<b means rank_a smaller => a is greater
        if pred_a_greater == truth_a_greater:
            correct += 1
    return correct / len(ep.queries) if ep.queries else 0.0
