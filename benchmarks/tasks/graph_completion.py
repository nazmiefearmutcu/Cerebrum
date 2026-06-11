import numpy as np
from cerebrum.types import Exogenous

def _goto_cell(net, cell):
    """Place the grid at the CANONICAL coordinate of `cell` via exogenous moves from origin.
    The grid is a non-wrapping planar path-integrator, so on a torus a cell must always be
    referenced by ONE canonical (unwrapped, in-range) coordinate; otherwise the same cell gets
    distinct grid codes at bind- vs query-time and completion fails (Task 14 worker note).
    The coordinate is set by the exogenous action stream / environment, never by network state."""
    net.grid.reset()
    net.move(Exogenous(np.array([cell[0], cell[1]], float)))

def run_cerebrum_episode(net, ep):
    """Walk the episode binding obs at each cell; then score held-out path-integrated completions."""
    # walk: at each step bind current obs at that cell's CANONICAL grid coordinate.
    # bind-pos for a cell == query-pos for the same cell (both the canonical coordinate),
    # which is the whole graph-completion mechanism.
    cell = (0,0)
    _goto_cell(net, cell)
    net.observe_and_learn(ep.gw.obs_at(cell), reward=1.0)
    for (c, a, avec) in ep.walk:
        cell = ep.gw.step(c, a)
        _goto_cell(net, cell)
        net.observe_and_learn(ep.gw.obs_at(cell), reward=1.0)
    # query: from start, path-integrate by displacement, complete, compare top-1 obs.
    # start + disp == target's canonical coordinate (disp is the torus-unwrapped displacement),
    # so the query grid code matches the bind-time code at the target cell.
    correct = 0
    for (start, disp, target) in ep.queries:
        net.grid.reset()
        # move grid to 'start' then by 'disp' (exogenous); start offset from origin (0,0)
        net.move(Exogenous(np.array([start[0], start[1]], float)))
        net.move(Exogenous(disp))
        pred = net.predict_obs_here(ep.gw.vocab)
        if pred.size and np.argmax(pred) == np.argmax(ep.gw.obs_at(target)):
            correct += 1
    return correct/len(ep.queries) if ep.queries else 0.0
