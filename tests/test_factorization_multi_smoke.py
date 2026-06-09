"""Smoke test for the MULTI-FACTOR factorization probe (benchmarks/run_factorization_multi.py).

CONTEXT — pushing the central-bet finding to a harder regime
------------------------------------------------------------
benchmarks/run_factorization.py established (corrected from an earlier degenerate null) that the
LOCAL four-factor plasticity builds a compositionally-generalizing FACTORED latent on a 2-factor
task: each factor is linearly decodable off the trained top latent on HELD-OUT combos well above
chance, above the untrained same-arch latent and a random-projection of the obs.

This probe (C1-MoreFactors) extends that to K = 3 (and 4) INDEPENDENT factors with larger
per-factor cardinality (up to ~6-8). The observation is a CONCAT of K frozen per-factor parts;
training is on a SUBSET of the (exponentially large) K-dim combo grid, every factor still probed
on HELD-OUT combos. The run script reports per-factor held-out decode (and the factor-average)
with CIs vs the untrained + random-projection controls + chance, as #factors / cardinality grow,
to map honestly where the local rule's factorization HOLDS vs BREAKS.

These assertions check only ROBUSTLY-TRUE contracts (finite/in-range outputs, split disjointness
+ coverage, determinism of the noise-free measurement, decoder sanity). They do NOT assert "the
latent factorizes at K=4" — that is the empirical question the run script answers honestly.
"""
import numpy as np

from benchmarks.run_factorization_multi import (
    MultiFactorTask, make_multi_split, settle_top_latent_multi,
    multi_factorization_probe, run_one_seed_multi,
)
from benchmarks.run_factorization import ncm_decode_acc, logistic_decode_acc


def test_multifactor_task_embed_is_concat_of_frozen_parts():
    # obs(c) must be the concatenation of K frozen per-factor parts, each depending ONLY on its
    # own factor value -> genuinely factorable input (the same contract as the 2-factor task).
    task = MultiFactorTask(cards=(4, 5, 6), part_dim=6, seed=0)
    assert task.K == 3
    assert task.obs_dim == 3 * 6
    e1 = task.embed(1, 2, 3)
    e2 = task.embed(1, 0, 0)  # share factor-0 value only
    # factor-0 slice must be identical (depends only on f0); other slices must differ
    s0 = task.slices[0]
    assert np.array_equal(e1[s0], e2[s0])
    assert not np.array_equal(e1[task.slices[1]], e2[task.slices[1]])
    # changing factor-0 alone changes ONLY the factor-0 slice
    e3 = task.embed(2, 2, 3)
    assert not np.array_equal(e1[s0], e3[s0])
    assert np.array_equal(e1[task.slices[1]], e3[task.slices[1]])
    assert np.array_equal(e1[task.slices[2]], e3[task.slices[2]])


def test_make_multi_split_disjoint_covers_every_value_and_caps_budget():
    cards = (6, 6, 6)
    train, held = make_multi_split(cards, n_combos=120, frac_heldout=0.3, seed=2)
    assert len(train) > 0 and len(held) > 0
    assert set(train).isdisjoint(set(held))
    # budget cap respected (full grid is 216; we asked for 120)
    assert len(train) + len(held) <= 120
    # every factor VALUE must still appear in training, so each held-out factor is decodable
    for k, card in enumerate(cards):
        seen = {c[k] for c in train}
        assert seen == set(range(card)), f"factor {k} missing values in train: {seen}"
    # held-out factor values are all seen in training (fair compositional probe)
    for c in held:
        for k in range(len(cards)):
            assert c[k] in {t[k] for t in train}


def test_settle_top_latent_multi_finite_and_deterministic():
    task = MultiFactorTask(cards=(4, 4, 4), part_dim=5, seed=0)
    from grail.config import GRAILConfig
    from grail.pc_core import PCAreas
    cfg = GRAILConfig(dims=(task.obs_dim, 20, 20), n_settle=8, seed=0)
    net = PCAreas(cfg)
    obs = task.embed(1, 2, 3)
    z1 = settle_top_latent_multi(net, obs, steps=10)
    z2 = settle_top_latent_multi(net, obs, steps=10)
    assert z1.shape == (cfg.dims[-1],)
    assert np.all(np.isfinite(z1))
    # T=0 noise-free readout is a pure function of weights -> bit-identical on repeat
    assert np.array_equal(z1, z2)


def test_decoders_handle_multiclass_separable_problem():
    # the measurement probes (shared with the 2-factor file) must score perfectly on a trivially
    # linearly-separable 6-class problem and never exceed [0,1].
    rng = np.random.default_rng(0)
    n_cls = 6
    Xtr = np.concatenate([np.full((4, 5), c) + 0.01 * rng.standard_normal((4, 5))
                          for c in range(n_cls)])
    ytr = np.repeat(np.arange(n_cls), 4)
    Xte = np.concatenate([np.full((2, 5), c) + 0.01 * rng.standard_normal((2, 5))
                          for c in range(n_cls)])
    yte = np.repeat(np.arange(n_cls), 2)
    a_ncm = ncm_decode_acc(Xtr, ytr, Xte, yte, n_cls)
    a_log = logistic_decode_acc(Xtr, ytr, Xte, yte, n_cls)
    assert 0.0 <= a_ncm <= 1.0 and 0.0 <= a_log <= 1.0
    assert a_ncm == 1.0 and a_log == 1.0


def test_multi_factorization_probe_returns_finite_per_factor_dict():
    task = MultiFactorTask(cards=(4, 4, 4), part_dim=5, seed=1)
    train, held = make_multi_split((4, 4, 4), n_combos=48, frac_heldout=0.3, seed=11)
    res = multi_factorization_probe(task, train, held, dims=(task.obs_dim, 20, 20),
                                    passes=4, seed=0, decoder="ncm")
    # one decode entry per factor per condition + the averages
    for cond in ("trained", "untrained", "randproj"):
        for k in range(task.K):
            key = f"{cond}_f{k}"
            assert key in res
            assert np.isfinite(res[key]) and 0.0 <= res[key] <= 1.0
        assert f"{cond}_avg" in res and np.isfinite(res[f"{cond}_avg"])
    assert res["n_train"] > 0 and res["n_held"] > 0


def test_run_one_seed_multi_has_both_probe_kinds_and_chance():
    out = run_one_seed_multi(seed=0, cards=(4, 4, 4), part_dim=5, width=20, depth=3,
                             n_combos=48, frac_heldout=0.3, passes=4)
    assert "ncm" in out and "logreg" in out
    assert "chance" in out and out["chance"] > 0
    for kind in ("ncm", "logreg"):
        for cond in ("trained", "untrained", "randproj"):
            for k in range(3):
                v = out[kind][f"{cond}_f{k}"]
                assert np.isfinite(v) and 0.0 <= v <= 1.0
