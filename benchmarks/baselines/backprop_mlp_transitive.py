"""Backprop-MLP baseline for TRANSITIVE INFERENCE (COMPARATOR ONLY — uses backprop,
which GRAIL never does).

Trained by gradient descent on the SHOWN adjacent pairs only: input = concat(one-hot
obs_i, one-hot obs_j) for an adjacent pair (both orderings, with the correct
"is-i-greater" label), output = P(i greater). Tested on held-out NON-adjacent pairs.

With only adjacent supervision and one-hot (orderless) item codes, the MLP has no
signal that lets it chain a transitive order — it can memorize the shown pairs but
must generalize to unseen pairs, where it sits near chance. This is the classic
result that a flat supervised learner needs the non-adjacent pairs (or an ordinal
input) to solve transitive inference.
"""
import numpy as np


def run_mlp_episode(ep, epochs=300, hidden=32, lr=0.2, seed=0):
    order = ep.order
    rng = np.random.default_rng(seed)
    V = order.vocab
    nin = 2 * V
    W1 = 0.1 * rng.standard_normal((hidden, nin)); b1 = np.zeros(hidden)
    w2 = 0.1 * rng.standard_normal(hidden); b2 = 0.0

    def feat(sym_i, sym_j):
        v = np.zeros(nin); v[sym_i] = 1.0; v[V + sym_j] = 1.0; return v

    # build training set from SHOWN adjacent pairs, both orderings
    X, Y = [], []
    for pair in ep.train_pairs:
        lo, hi = sorted(pair)               # lo rank => greater
        sg, sl = order.symbol[lo], order.symbol[hi]
        X.append(feat(sg, sl)); Y.append(1.0)   # i=greater -> label 1
        X.append(feat(sl, sg)); Y.append(0.0)   # i=lesser  -> label 0
    X = np.array(X); Y = np.array(Y)
    if X.shape[0] == 0:
        return 0.0

    for _ in range(epochs):
        h = np.tanh(X @ W1.T + b1)
        z = h @ w2 + b2
        p = 1.0 / (1.0 + np.exp(-z))
        g = (p - Y) / len(X)                     # dL/dz, logistic
        gw2 = h.T @ g; gb2 = g.sum()
        gh = np.outer(g, w2) * (1 - h ** 2)
        gW1 = gh.T @ X; gb1 = gh.sum(0)
        w2 -= lr * gw2; b2 -= lr * gb2; W1 -= lr * gW1; b1 -= lr * gb1

    correct = 0
    for (a, b) in ep.queries:
        sa, sb = order.symbol[a], order.symbol[b]
        h = np.tanh(feat(sa, sb) @ W1.T + b1); z = h @ w2 + b2
        pred_a_greater = (z > 0.0)
        truth_a_greater = (a < b)
        if pred_a_greater == truth_a_greater:
            correct += 1
    return correct / len(ep.queries) if ep.queries else 0.0
