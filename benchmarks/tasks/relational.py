"""NON-METRIC / ASYMMETRIC relational few-shot benchmark  (spec failure-mode FM7).

This is the adversarial sibling of Task-1 (benchmarks/tasks/gridworld.py +
graph_completion.py). Task-1 lives on a torus gridworld: a genuinely METRIC space where
the four actions are 2D displacement vectors that COMMUTE and where the position of any
cell is a linear function of the actions taken to reach it. GRAIL's grid HEAD path-
integrates exactly that algebra (`pos += action.value`; code = [cos(k.pos), sin(k.pos)]),
so two different paths to the same cell land on the same grid code and content-completion
recalls the right observation. That is why grid beats flat there.

Here we instead build a random DIRECTED graph:

  * `n_nodes` abstract nodes, each carrying a fixed one-hot observation (structure-free
    content, exactly like gridworld's per-cell object id).
  * `n_relations` relation types. Each relation r is a *deterministic but arbitrary*
    permutation-like map node -> node (`_succ[r]`), built from a random RNG with NO 2D
    geometry behind it. a --r--> b does NOT imply b --r--> a (ASYMMETRIC), and the maps
    for different relations do NOT commute (`step(step(a,r1),r2) != step(step(a,r2),r1)`
    in general). There is no Euclidean embedding in which "follow r" is a fixed
    displacement.

  * Crucially, GRAIL is still only allowed an EXOGENOUS action label. We hand each
    relation r a FROZEN, arbitrary 2D vector `relation_vec(r)` (the only thing a metric
    path-integrator can consume) and feed it as Exogenous(...). This vector is a pure
    external label of the relation id — it is NOT a function of node/obs/network state
    (BAN-1 respected). The grid will *assume* these vectors add up metrically; on this
    graph they do not correspond to node identity, which is the whole point.

Mechanistic prediction: when a query node is reached by composing relations along a path
the agent did not walk as a single bound step, the grid's summed relation-vector
(commutative, path-collapsing) will generically NOT equal the relation-vector sum that was
in effect when the target node was bound during the walk — because node identity on a
non-metric graph is not a linear function of relation vectors. So the grid code at query
time mismatches the bind-time code and content-completion recalls the WRONG observation.
The grid prior should therefore lose its Task-1 advantage and collapse toward (or below)
the flat prior. Reporting that honestly is the deliverable.
"""
import numpy as np
from dataclasses import dataclass
from grail.types import Exogenous


class RelationalGraph:
    """Random directed graph with asymmetric, non-commuting relations and per-node obs."""

    def __init__(self, n_nodes, n_relations, vocab, seed=0):
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.vocab = vocab
        rng = np.random.default_rng(seed)
        # per-node one-hot observation content (no structure, just like gridworld._obj)
        self._obj = rng.integers(0, vocab, size=n_nodes)
        # successor table: _succ[r][a] = node reached by following relation r out of a.
        # Built as a random self-map per relation (NOT a clean permutation, NOT symmetric,
        # NOT geometric) -> genuinely non-metric reachability.
        self._succ = np.stack(
            [rng.integers(0, n_nodes, size=n_nodes) for _ in range(n_relations)], axis=0
        )
        # FROZEN arbitrary 2D label vector per relation id (the only thing a metric grid
        # path-integrator can consume). Pure external label of the relation; never derived
        # from node/obs/state. Drawn from a separate RNG stream so it is uncorrelated with
        # the actual (non-metric) successor structure.
        rng_v = np.random.default_rng(seed + 4242)
        self._relvec = rng_v.standard_normal((n_relations, 2))

    def obs_at(self, node):
        v = np.zeros(self.vocab)
        v[self._obj[node % self.n_nodes]] = 1.0
        return v

    def step(self, node, relation):
        """Follow directed relation `relation` out of `node` (asymmetric, non-metric)."""
        return int(self._succ[relation, node % self.n_nodes])

    def relation_vec(self, relation):
        """Frozen external 2D label for a relation id (fed to the grid as Exogenous)."""
        return self._relvec[relation].copy()


@dataclass
class Episode:
    g: RelationalGraph
    walk: list           # list of (node, relation_id, relation_vec)
    observed_nodes: set
    queries: list        # (start_node, rel_path[list of relation_ids], target_node)


def make_episode(n_nodes, n_relations, vocab, K, seed=0):
    """Walk K random directed relation-steps; bind obs at each visited node.

    Held-out queries are 2-hop relation COMPOSITIONS r1 then r2 such that:
      - both r1 and r2 individually appear somewhere in the walk (relations are 'seen'),
      - the composed path (start --r1--> mid --r2--> target) is NOT itself a single
        walked edge,
      - the target node was observed during the walk (so its obs is known / scorable).
    The agent must therefore predict obs at `target` by COMPOSING relations it has seen,
    exactly the few-shot relational-reasoning ask. On a metric graph this is what path
    integration nails; on this non-metric graph it should fail.
    """
    g = RelationalGraph(n_nodes, n_relations, vocab, seed=seed)
    rng = np.random.default_rng(seed + 1)
    node = 0
    walk = []
    observed = {node}
    walked_single_edges = set()         # (start_node, relation_id) actually traversed
    seen_relations = set()
    for _ in range(K):
        r = int(rng.integers(0, n_relations))
        nxt = g.step(node, r)
        walk.append((node, r, g.relation_vec(r)))
        walked_single_edges.add((node, r))
        seen_relations.add(r)
        node = nxt
        observed.add(node)

    obs_list = sorted(observed)
    seen_rel_list = sorted(seen_relations)
    queries = []
    # enumerate 2-hop compositions from each observed start over seen relations
    for s in obs_list:
        for r1 in seen_rel_list:
            mid = g.step(s, r1)
            for r2 in seen_rel_list:
                target = g.step(mid, r2)
                if target not in observed:
                    continue                       # need a known obs to score against
                if (s, r1) in walked_single_edges and g.step(s, r1) == target:
                    continue                       # would be a trivially-walked single step
                queries.append((s, [r1, r2], target))
    # dedup + cap (deterministic order)
    seen_q = set()
    uniq = []
    for q in queries:
        key = (q[0], tuple(q[1]), q[2])
        if key in seen_q:
            continue
        seen_q.add(key)
        uniq.append(q)
    return Episode(g=g, walk=walk, observed_nodes=observed, queries=uniq[:64])


def _goto_node_origin(net):
    """Reset the grid to the origin. Node identity is set ONLY by the exogenous relation
    stream applied afterward (never by network state)."""
    net.grid.reset()


def run_grail_episode(net, ep):
    """Bind obs along the walk at the grid code produced by the cumulative relation-vector
    sum; then score held-out 2-hop relational completions.

    The grid code for a node = [cos(k . sum_of_relation_vecs), sin(...)] where the sum is
    over the FROZEN exogenous relation vectors applied to reach that node from the origin.
    On a metric graph this cumulative sum is a faithful position; on this non-metric graph
    the same node reached by different relation paths gets DIFFERENT sums (and different
    relation paths to a node are the rule, not the exception), so bind-time and query-time
    codes for the target node disagree and completion recalls the wrong obs.
    """
    g = ep.g
    # ---- bind phase: walk the graph, advancing grid by each relation's exogenous vector
    net.grid.reset()
    node = 0
    net.observe_and_learn(g.obs_at(node), reward=1.0)        # bind start obs at origin code
    for (n, r, rvec) in ep.walk:
        net.move(Exogenous(rvec))                            # exogenous relation label only
        node = g.step(n, r)
        net.observe_and_learn(g.obs_at(node), reward=1.0)    # bind obs at the new grid code
    # ---- query phase: from origin, path-integrate start's first-seen cumulative offset?
    # We do NOT have a canonical per-node coordinate here (that is exactly what a non-metric
    # graph denies us). The fair, honest analogue of Task-1 is: replay the relation path the
    # query specifies, starting from the grid code where `start` was bound, by composing the
    # exogenous relation vectors, then complete. We approximate start's bind code by the
    # cumulative relation-vector sum at its FIRST occurrence in the walk (the agent's own
    # episodic anchor for that node).
    first_offset = {}                                        # node -> cumulative relvec sum at first visit
    cum = np.zeros(2)
    first_offset.setdefault(0, cum.copy())
    for (n, r, rvec) in ep.walk:
        cum = cum + rvec
        nxt = g.step(n, r)
        if nxt not in first_offset:
            first_offset[nxt] = cum.copy()
    correct = 0
    for (start, rel_path, target) in ep.queries:
        net.grid.reset()
        # move grid to start's episodic anchor (exogenous cumulative offset), then compose
        # the queried relation vectors (still exogenous) — pure metric path-integration.
        net.move(Exogenous(first_offset.get(start, np.zeros(2))))
        for r in rel_path:
            net.move(Exogenous(g.relation_vec(r)))
        pred = net.predict_obs_here(g.vocab)
        if pred.size and np.argmax(pred) == np.argmax(g.obs_at(target)):
            correct += 1
    return correct / len(ep.queries) if ep.queries else 0.0


class HierarchyGraph:
    """Directed tree/hierarchy graph using heap-like binary tree indexing and per-node obs."""

    def __init__(self, n_nodes, vocab, seed=0):
        self.n_nodes = n_nodes
        self.n_relations = 3
        self.vocab = vocab
        rng = np.random.default_rng(seed)
        # per-node one-hot observation content
        self._obj = rng.integers(0, vocab, size=n_nodes)
        
        # FROZEN arbitrary 2D label vector per relation id (0=parent, 1=child_left, 2=child_right).
        rng_v = np.random.default_rng(seed + 4242)
        self._relvec = rng_v.standard_normal((3, 2))

    def obs_at(self, node):
        v = np.zeros(self.vocab)
        v[self._obj[node % self.n_nodes]] = 1.0
        return v

    def step(self, node, relation):
        """Follow relation (0 = parent, 1 = child_left, 2 = child_right) out of node."""
        i = int(node % self.n_nodes)
        rel = int(relation)
        if rel == 0:
            return (i - 1) // 2 if i > 0 else 0
        elif rel == 1:
            left = 2 * i + 1
            return left if left < self.n_nodes else i
        elif rel == 2:
            right = 2 * i + 2
            return right if right < self.n_nodes else i
        else:
            raise ValueError(f"Invalid relation {relation}")

    def relation_vec(self, relation):
        """Frozen external 2D label for a relation id (fed to the grid as Exogenous)."""
        return self._relvec[relation].copy()


def make_hierarchy_episode(n_nodes, vocab, K, seed=0):
    """Walk K random directed relation-steps on hierarchy; bind obs at each visited node."""
    g = HierarchyGraph(n_nodes, vocab, seed=seed)
    rng = np.random.default_rng(seed + 1)
    node = 0
    walk = []
    observed = {node}
    walked_single_edges = set()         # (start_node, relation_id) actually traversed
    seen_relations = set()
    n_relations = 3
    for _ in range(K):
        r = int(rng.integers(0, n_relations))
        nxt = g.step(node, r)
        walk.append((node, r, g.relation_vec(r)))
        walked_single_edges.add((node, r))
        seen_relations.add(r)
        node = nxt
        observed.add(node)

    obs_list = sorted(observed)
    seen_rel_list = sorted(seen_relations)
    queries = []
    # enumerate 2-hop compositions from each observed start over seen relations
    for s in obs_list:
        for r1 in seen_rel_list:
            mid = g.step(s, r1)
            for r2 in seen_rel_list:
                target = g.step(mid, r2)
                if target not in observed:
                    continue                       # need a known obs to score against
                if (s, r1) in walked_single_edges and g.step(s, r1) == target:
                    continue                       # would be a trivially-walked single step
                queries.append((s, [r1, r2], target))
    # dedup + cap (deterministic order)
    seen_q = set()
    uniq = []
    for q in queries:
        key = (q[0], tuple(q[1]), q[2])
        if key in seen_q:
            continue
        seen_q.add(key)
        uniq.append(q)
    return Episode(g=g, walk=walk, observed_nodes=observed, queries=uniq[:64])

