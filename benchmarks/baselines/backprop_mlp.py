import numpy as np

def run_mlp_episode(ep, epochs=100, hidden=32, lr=0.1, seed=0):
    """Baseline COMPARATOR ONLY (uses backprop — GRAIL never does). Maps start-cell-onehot + disp -> obs."""
    rng = np.random.default_rng(seed)
    cells = sorted(ep.observed_cells); idx = {c:i for i,c in enumerate(cells)}
    nin = len(cells) + 2; nout = ep.gw.vocab
    W1 = 0.1*rng.standard_normal((hidden, nin)); b1 = np.zeros(hidden)
    W2 = 0.1*rng.standard_normal((nout, hidden)); b2 = np.zeros(nout)
    def feat(start, disp):
        v = np.zeros(nin); v[idx[start]] = 1.0; v[-2:] = disp; return v
    # train on WALKED edges only (few-shot supervision)
    Xtr = []; Ytr = []
    cell = (0,0)
    for (c,a,avec) in ep.walk:
        nxt = ep.gw.step(c,a); Xtr.append(feat(c, avec)); Ytr.append(ep.gw.obs_at(nxt))
    Xtr = np.array(Xtr); Ytr = np.array(Ytr)
    for _ in range(epochs):
        h = np.tanh(Xtr@W1.T + b1); logits = h@W2.T + b2
        p = np.exp(logits - logits.max(1,keepdims=True)); p /= p.sum(1,keepdims=True)
        g = (p - Ytr)/len(Xtr)
        gW2 = g.T@h; gb2 = g.sum(0); gh = (g@W2)*(1-h**2)
        gW1 = gh.T@Xtr; gb1 = gh.sum(0)
        W2 -= lr*gW2; b2 -= lr*gb2; W1 -= lr*gW1; b1 -= lr*gb1
    correct = 0
    for (start, disp, target) in ep.queries:
        x = feat(start, disp); h = np.tanh(x@W1.T+b1); logits = h@W2.T+b2
        if np.argmax(logits) == np.argmax(ep.gw.obs_at(target)): correct += 1
    return correct/len(ep.queries) if ep.queries else 0.0
