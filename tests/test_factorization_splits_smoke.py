"""Smoke test for the C2-HardSplits factorization probe (benchmarks/run_factorization_splits.py).

CONTEXT — systematic vs interpolative compositional generalization
-------------------------------------------------------------------
benchmarks/run_factorization.py (§g) and run_factorization_multi.py (§g2, C1-MoreFactors)
established that the LOCAL four-factor plasticity builds a compositionally-generalizing FACTORED
latent under a RANDOM held-out split of the combo grid: each factor is linearly decodable off the
trained top latent x[top] on held-out combos, above the untrained same-arch latent and (at small
cardinality) above a random-projection of the obs.

A random held-out split tests INTERPOLATION: every held-out factor value appears in MANY training
combos, so the latent need only interpolate within a densely-sampled grid. C2-HardSplits tests
SYSTEMATIC compositional generalization with HARDER hold-out structure:

  (a) LEAVE-A-VALUE-IN-FEW-CONTEXTS — a particular factor value appears in training in only 1-2
      combinations; test decoding/using it in NEW contexts (does the factor representation
      generalize from SPARSE context exposure?).
  (b) ROW / COLUMN hold-out (PRODUCTIVITY) — hold out a whole structured region of the combo grid
      (a whole row = a fixed value of one factor crossed with every value of the other) and test
      decode there.

The run script reports held-out decode under these hard splits vs the random split vs the
untrained + random-projection controls (CIs over seeds), and renders a SYSTEMATIC vs INTERPOLATE
verdict with the mechanism. The split builders enforce decodability-in-principle invariants
(every held-out factor VALUE still appears in training somewhere) so a null is informative, not a
trivially-unsolvable artifact.

These assertions check only ROBUSTLY-TRUE contracts (split disjointness + coverage + the
defining structure of each split, finite/in-range decode outputs, determinism of the noise-free
measurement). They do NOT assert "decode holds under hard splits" — that is the empirical
question the run script answers honestly.
"""
import numpy as np

from benchmarks.run_factorization_multi import MultiFactorTask
from benchmarks.run_factorization_splits import (
    make_random_split, make_few_context_split, make_row_split,
    split_probe, run_one_seed_splits, SPLIT_KINDS,
)


# ------------------------------------------------------------------------------------------
# Split builders: structure + decodability-in-principle invariants
# ------------------------------------------------------------------------------------------

def _covers_every_value(train, cards):
    return all({c[k] for c in train} == set(range(cards[k])) for k in range(len(cards)))


def test_random_split_disjoint_and_covers_every_value():
    cards = (6, 6)
    train, held, meta = make_random_split(cards, frac_heldout=0.3, seed=3)
    assert len(train) > 0 and len(held) > 0
    assert set(train).isdisjoint(set(held))
    assert _covers_every_value(train, cards)
    # every held-out factor value is still seen in training (fair compositional probe)
    for c in held:
        for k in range(len(cards)):
            assert c[k] in {t[k] for t in train}


def test_few_context_split_makes_every_target_value_globally_sparse():
    # the defining structure (global-sparse): EVERY value of the target factor appears in exactly
    # n_contexts TRAIN combos, and the held-out set is all the OTHER (new-context) combos. Every
    # target value MUST still be seen in training (decodable), just sparsely; co-factor coverage is
    # repaired so no other value is unseen.
    cards = (6, 6)
    for n_contexts in (1, 2):
        train, held, meta = make_few_context_split(cards, n_contexts=n_contexts,
                                                    target_factor=0, seed=1)
        assert set(train).isdisjoint(set(held))
        assert len(held) > 0
        # EVERY target value appears in EXACTLY n_contexts train combos (before coverage repair may
        # add a FEW more for co-factor coverage, so assert >= n_contexts and that the typical case
        # is exactly n_contexts for most values).
        from collections import Counter
        cnt = Counter(c[0] for c in train)
        assert set(cnt) == set(range(6))              # every target value seen
        assert all(v >= n_contexts for v in cnt.values())
        assert min(cnt.values()) == n_contexts        # at least one value at the sparse floor
        # every factor value still appears in training (coverage repair guarantees this)
        assert _covers_every_value(train, cards)
        assert meta["target_factor"] == 0 and meta["kind"] == "few_context"
        assert meta["n_contexts"] == n_contexts


def test_row_split_holds_out_a_structured_rectangular_block():
    # ROW/BLOCK hold-out: hold out a structured rectangle = a block of target-factor rows crossed
    # with a set of co-factor columns. PRODUCTIVITY test (structured region, not random scatter).
    # Each held row value must still appear in training (NON-held columns) and each held column
    # value via NON-held rows -> every value decodable in principle.
    cards = (6, 6)
    train, held, meta = make_row_split(cards, target_factor=0, target_value=2,
                                       block_rows=2, n_held_in_row=4, seed=2)
    assert set(train).isdisjoint(set(held))
    held_rows = set(meta["held_rows"])
    assert all(c[0] in held_rows for c in held)        # entire held region is in the row block
    assert len(held) == 2 * 4                           # block_rows x held columns
    # each held row value still appears in training (its non-held columns) -> decodable
    for r in held_rows:
        assert r in {t[0] for t in train}
    # coverage of every value preserved (block_rows < card so non-held rows cover held columns)
    assert _covers_every_value(train, cards)
    # the held columns (f1 values) appear in training via NON-held rows
    for c in held:
        assert c[1] in {t[1] for t in train}


def test_split_kinds_constant_lists_the_three_splits():
    assert set(SPLIT_KINDS) == {"random", "few_context", "row"}


# ------------------------------------------------------------------------------------------
# Probe: finite / in-range / per-condition contract
# ------------------------------------------------------------------------------------------

def test_split_probe_returns_finite_in_range_per_condition():
    task = MultiFactorTask(cards=(5, 5), part_dim=6, seed=1)
    train, held, meta = make_random_split((5, 5), frac_heldout=0.3, seed=11)
    res = split_probe(task, train, held, dims=(task.obs_dim, 18, 18), passes=4, seed=0,
                      decoder="ncm")
    for cond in ("trained", "untrained", "randproj"):
        # the headline metric is decode of the TARGET factor on held-out (for few_context/row that
        # is the sparse / held factor; for random it averages over factors)
        key = f"{cond}_target"
        assert key in res and np.isfinite(res[key]) and 0.0 <= res[key] <= 1.0
    assert res["n_train"] > 0 and res["n_held"] > 0


def test_run_one_seed_splits_runs_all_three_kinds_with_both_probes():
    out = run_one_seed_splits(seed=0, cards=(5, 5), part_dim=6, width=18, depth=3,
                              passes=4)
    for kind in SPLIT_KINDS:
        assert kind in out
        assert "chance" in out[kind] and out[kind]["chance"] > 0
        for probe in ("ncm", "logreg"):
            for cond in ("trained", "untrained", "randproj"):
                v = out[kind][probe][f"{cond}_target"]
                assert np.isfinite(v) and 0.0 <= v <= 1.0


def test_few_context_decode_is_deterministic_noise_free():
    # the whole pipeline is a pure function of (weights, seed): T=0 noise-free settle + fixed-seed
    # decoders -> identical numbers on repeat (guards against accidental nondeterminism creeping
    # into the measurement).
    task = MultiFactorTask(cards=(5, 5), part_dim=6, seed=2)
    train, held, meta = make_few_context_split((5, 5), n_contexts=2, target_factor=0,
                                               target_value=1, seed=7)
    a = split_probe(task, train, held, dims=(task.obs_dim, 18, 18), passes=4, seed=0,
                    decoder="ncm")
    b = split_probe(task, train, held, dims=(task.obs_dim, 18, 18), passes=4, seed=0,
                    decoder="ncm")
    assert a["trained_target"] == b["trained_target"]
    assert a["untrained_target"] == b["untrained_target"]
