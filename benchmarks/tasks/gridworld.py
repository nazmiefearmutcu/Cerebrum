import numpy as np
from dataclasses import dataclass

ACTIONS = {"N":(-1,0), "S":(1,0), "E":(0,1), "W":(0,-1)}

class GridWorld:
    def __init__(self, h, w, vocab, seed=0):
        self.h, self.w, self.vocab = h, w, vocab
        rng = np.random.default_rng(seed)
        self._obj = rng.integers(0, vocab, size=(h,w))    # object id per cell (structure-free content)
    def obs_at(self, cell):
        r, c = cell; v = np.zeros(self.vocab); v[self._obj[r % self.h, c % self.w]] = 1.0; return v
    def step(self, cell, action_name):
        dr, dc = ACTIONS[action_name]; return ((cell[0]+dr) % self.h, (cell[1]+dc) % self.w)

@dataclass
class Episode:
    gw: GridWorld
    walk: list           # list of (cell, action_name, action_vec)
    observed_cells: set
    queries: list        # (start_cell, displacement_vec, target_cell)

def make_episode(h, w, vocab, K, seed=0):
    gw = GridWorld(h, w, vocab, seed=seed); rng = np.random.default_rng(seed+1)
    names = list(ACTIONS); cell = (0,0); walk = []; observed = {cell}
    walked_edges = set()
    for _ in range(K):
        a = names[rng.integers(0,len(names))]; nxt = gw.step(cell, a)
        walk.append((cell, a, np.array(ACTIONS[a], float)))
        walked_edges.add((cell, nxt)); cell = nxt; observed.add(cell)
    # held-out queries: pairs of observed cells whose connecting straight path was NOT a walked edge
    obs_list = sorted(observed); queries = []
    for s in obs_list:
        for t in obs_list:
            if s == t: continue
            disp = np.array([(t[0]-s[0]), (t[1]-s[1])], float)  # raw displacement (torus-unwrapped)
            if (s, t) not in walked_edges:
                queries.append((s, disp, t))
    return Episode(gw=gw, walk=walk, observed_cells=observed, queries=queries[:64])
