"""Smoke test for the PRINCIPLED factorization probe (benchmarks/run_factorization.py).

CONTEXT — correcting an overstated negative
-------------------------------------------
The earlier compositional "f1->f2 completion" probe concluded the local plasticity does NOT
build compositional structure (a NULL across depth). A follow-up diagnosis showed that NULL was
largely a DEGENERATE-TASK ARTIFACT: f1 and f2 are INDEPENDENT factors, and the completion target
was the systematically-EXCLUDED f2 for each f1, so predicting it from f1 alone is
information-theoretically IMPOSSIBLE (a backprop-MLP and a memorizer fail it too). The SAME
diagnosis found the trained latent LINEARLY DECODES both f1 and f2 well above chance — i.e. the
local rule DOES represent the factors.

This probe is the principled, non-degenerate test: fit a LINEAR readout (a measurement probe,
NOT part of CEREBRUM) on the trained latent x[top] over SEEN combos and evaluate factor-decoding
accuracy on HELD-OUT combos. That is genuine compositional generalization: can each factor be
read off the latent for combinations never trained?

These assertions only check ROBUSTLY-TRUE things (finite/in-range outputs, shape/coverage
contracts, determinism of the noise-free measurement, the linear probe behaves sanely). They do
NOT assert "the latent factorizes" — that is the empirical question the run script answers.
"""
import numpy as np

from cerebrum.config import CerebrumConfig
from benchmarks.tasks.compositional import CompositionalTask
from benchmarks.run_factorization import (
    make_split, settle_top_latent, ncm_decode_acc, logistic_decode_acc,
    factorization_probe, run_one_seed,
)


def test_make_split_disjoint_and_covers_every_factor_value():
    # a fair compositional split must (a) be disjoint, (b) leave every factor VALUE in training
    # so each held-out factor is in principle decodable, (c) actually hold something out.
    train, held = make_split(A=5, B=5, frac_heldout=0.3, seed=3)
    assert len(train) > 0 and len(held) > 0
    assert set(train).isdisjoint(set(held))
    seen_f1 = {f1 for (f1, _) in train}
    seen_f2 = {f2 for (_, f2) in train}
    assert seen_f1 == set(range(5)) and seen_f2 == set(range(5))
    for (f1, f2) in held:
        assert f1 in seen_f1 and f2 in seen_f2


def test_settle_top_latent_shape_finite_and_deterministic():
    task = CompositionalTask(A=4, B=4, part_dim=6, seed=0)
    from cerebrum.pc_core import PCAreas
    cfg = CerebrumConfig(dims=(task.obs_dim, 16, 16), n_settle=8, seed=0)
    net = PCAreas(cfg)
    obs = task.embed(1, 2)
    z1 = settle_top_latent(net, obs, steps=10)
    z2 = settle_top_latent(net, obs, steps=10)
    assert z1.shape == (cfg.dims[-1],)
    assert np.all(np.isfinite(z1))
    # T=0 noise-free readout is a pure function of weights -> bit-identical on repeat
    assert np.array_equal(z1, z2)


def test_ncm_and_logistic_decoders_recover_a_linearly_separable_factor():
    # build a trivially linearly-separable 2-class problem; both probes must score perfectly,
    # and never below chance. (sanity on the MEASUREMENT probes themselves, not on CEREBRUM.)
    rng = np.random.default_rng(0)
    n_cls = 3
    Xtr = np.concatenate([np.full((6, 4), c) + 0.01 * rng.standard_normal((6, 4))
                          for c in range(n_cls)])
    ytr = np.repeat(np.arange(n_cls), 6)
    Xte = np.concatenate([np.full((2, 4), c) + 0.01 * rng.standard_normal((2, 4))
                          for c in range(n_cls)])
    yte = np.repeat(np.arange(n_cls), 2)
    a_ncm = ncm_decode_acc(Xtr, ytr, Xte, yte, n_cls)
    a_log = logistic_decode_acc(Xtr, ytr, Xte, yte, n_cls)
    assert 0.0 <= a_ncm <= 1.0 and 0.0 <= a_log <= 1.0
    assert a_ncm == 1.0 and a_log == 1.0


def test_factorization_probe_returns_finite_in_range_dict():
    task = CompositionalTask(A=4, B=4, part_dim=6, seed=1)
    train, held = make_split(A=4, B=4, frac_heldout=0.3, seed=11)
    res = factorization_probe(task, train, held, dims=(task.obs_dim, 16, 16),
                              passes=6, seed=0)
    for k in ("trained_f1", "trained_f2", "untrained_f1", "untrained_f2",
              "raw_f1", "raw_f2", "randproj_f1", "randproj_f2"):
        assert k in res
        assert np.isfinite(res[k]) and 0.0 <= res[k] <= 1.0


def test_run_one_seed_finite_and_has_both_probe_kinds():
    out = run_one_seed(seed=0, A=4, B=4, part_dim=6, width=16, depth=3,
                       frac_heldout=0.3, passes=6)
    # both the nearest-class-mean and logistic measurement probes are reported
    assert "ncm" in out and "logreg" in out
    for kind in ("ncm", "logreg"):
        for k in ("trained_f1", "trained_f2", "untrained_f1", "untrained_f2",
                  "raw_f1", "raw_f2", "randproj_f1", "randproj_f2"):
            assert np.isfinite(out[kind][k]) and 0.0 <= out[kind][k] <= 1.0
