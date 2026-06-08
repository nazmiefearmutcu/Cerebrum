"""Flat-prior comparator for the NON-METRIC relational task (mirror of baselines/flat_prior.py).

Random fixed code per VISITED node, no path integration / no transition algebra. It can
only recall an observation at a code it has actually bound; a composed relational query
cannot be synthesized, so it answers from the START node's code -> generally wrong target.
This is the structure-free reference: anything a *useful* structured prior buys must beat
this.
"""
import numpy as np


def run_flat_relational_episode(ep):
    rng = np.random.default_rng(123)
    code = {}
    g = ep.g

    def code_of(node):
        if node not in code:
            code[node] = rng.standard_normal(16)
        return code[node]

    node = 0
    store = np.zeros((g.vocab, 16))
    store += np.outer(g.obs_at(node), code_of(node))
    for (n, r, rvec) in ep.walk:
        node = g.step(n, r)
        store += np.outer(g.obs_at(node), code_of(node))

    correct = 0
    for (start, rel_path, target) in ep.queries:
        # no relational algebra: best it can do is recall at the start node's bound code
        pred = store @ code_of(start)
        if np.argmax(pred) == np.argmax(g.obs_at(target)):
            correct += 1
    return correct / len(ep.queries) if ep.queries else 0.0
