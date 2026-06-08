import numpy as np

def run_flat_episode(ep):
    """Flat prior: random fixed code per VISITED cell, no path integration. Recall only at bound codes."""
    rng = np.random.default_rng(123)
    code = {}; store = None
    def code_of(cell):
        if cell not in code: code[cell] = rng.standard_normal(16)
        return code[cell]
    cell = (0,0); store = np.zeros((ep.gw.vocab, 16))
    store += np.outer(ep.gw.obs_at(cell), code_of(cell))
    for (c, a, avec) in ep.walk:
        cell = ep.gw.step(c, a); store += np.outer(ep.gw.obs_at(cell), code_of(cell))
    correct = 0
    for (start, disp, target) in ep.queries:
        # flat prior has NO transition algebra: a path-integrated query cannot synthesize target's code;
        # it can only guess from the start cell's code -> wrong target obs.
        pred = store @ code_of(start)
        if np.argmax(pred) == np.argmax(ep.gw.obs_at(target)): correct += 1
    return correct/len(ep.queries) if ep.queries else 0.0
