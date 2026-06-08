"""Backprop-MLP comparator for the NON-METRIC relational task.

COMPARATOR ONLY — uses gradient descent / backprop, which GRAIL never does. Kept in its own
file so the GRAIL pathway and the existing baselines/backprop_mlp.py are untouched. A small
2-layer tanh net is trained on the WALKED single relation-steps (few-shot supervision):
features = one-hot(start_node) concatenated with the relation's frozen 2D label vector;
target = obs at the successor node. At query time the 2-hop composition is applied node-wise
in feature space (predict mid from start+r1, take argmax->mid-onehot is not available, so we
instead feed the TRUE composed structure the same way Task-1's MLP does: one-hot(start) + the
summed relation label vector for the path). Because a free-form learner CAN in principle
memorize an arbitrary node->node table from enough examples, this baseline shows what a
non-structured but trainable model achieves under the same few-shot budget.

No autograd library: gradients are written by hand (same as baselines/backprop_mlp.py).
"""
import numpy as np


def run_mlp_relational_episode(ep, epochs=100, hidden=32, lr=0.1, seed=0):
    rng = np.random.default_rng(seed)
    g = ep.g
    nodes = sorted(ep.observed_nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    nin = len(nodes) + 2
    nout = g.vocab
    W1 = 0.1 * rng.standard_normal((hidden, nin)); b1 = np.zeros(hidden)
    W2 = 0.1 * rng.standard_normal((nout, hidden)); b2 = np.zeros(nout)

    def feat(start, relvec_sum):
        v = np.zeros(nin)
        v[idx[start]] = 1.0
        v[-2:] = relvec_sum
        return v

    # train on walked single relation-steps (start_node + relation label vec -> successor obs)
    Xtr = []; Ytr = []
    for (n, r, rvec) in ep.walk:
        nxt = g.step(n, r)
        if n in idx:
            Xtr.append(feat(n, rvec))
            Ytr.append(g.obs_at(nxt))
    if not Xtr:
        return 0.0
    Xtr = np.array(Xtr); Ytr = np.array(Ytr)
    for _ in range(epochs):
        h = np.tanh(Xtr @ W1.T + b1)
        logits = h @ W2.T + b2
        p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        gr = (p - Ytr) / len(Xtr)
        gW2 = gr.T @ h; gb2 = gr.sum(0)
        gh = (gr @ W2) * (1 - h ** 2)
        gW1 = gh.T @ Xtr; gb1 = gh.sum(0)
        W2 -= lr * gW2; b2 -= lr * gb2; W1 -= lr * gW1; b1 -= lr * gb1

    correct = 0
    for (start, rel_path, target) in ep.queries:
        if start not in idx:
            continue
        relvec_sum = np.sum([g.relation_vec(r) for r in rel_path], axis=0)
        x = feat(start, relvec_sum)
        h = np.tanh(x @ W1.T + b1)
        logits = h @ W2.T + b2
        if np.argmax(logits) == np.argmax(g.obs_at(target)):
            correct += 1
    return correct / len(ep.queries) if ep.queries else 0.0
