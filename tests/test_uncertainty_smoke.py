"""Fast smoke test for the Pillar-4 (Langevin) uncertainty-quantification benchmark.

We only assert robustly-true facts: finite/bounded outputs, that distinct noise seeds DO
produce a genuine sample distribution (real disagreement, not bit-identical reruns), and the
AUROC/metric plumbing is well-formed. We deliberately do NOT assert that the uncertainty is
"calibrated" -- that is the open scientific question the benchmark reports on, and the effect
is only weak, so a hard calibration assertion would be flaky and dishonest.
"""
import numpy as np
from grail.config import GRAILConfig
from grail.rng import SeededRNG
from benchmarks.tasks.gridworld import make_episode
from benchmarks.run_uncertainty import (
    train_episode, settle_samples, per_query_records,
    calibration_for_seed, auroc,
)


def _small_cfg(vocab=5, seed=0):
    return GRAILConfig(dims=(vocab, 8, 8), grid_n_modules=8, n_settle=10, seed=seed)


def test_settle_samples_finite_and_shaped():
    ep = make_episode(h=4, w=4, vocab=5, K=10, seed=1)
    cfg = _small_cfg(seed=1)
    net = train_episode(ep, cfg)
    start, disp, _target = ep.queries[0]
    comp, P = settle_samples(net, cfg, start, disp, S=8, n_settle=15, T=cfg.T_floor)
    assert comp.shape == (5,)
    assert P.shape == (8, 5)
    assert np.all(np.isfinite(comp))
    assert np.all(np.isfinite(P))


def test_langevin_noise_produces_real_sample_spread():
    # Distinct seeds must give genuinely different settles (Pillar 4: T_floor > 0 forbids collapse).
    ep = make_episode(h=4, w=4, vocab=5, K=12, seed=2)
    cfg = _small_cfg(seed=0)
    assert cfg.T_floor > 0.0
    net = train_episode(ep, cfg)
    start, disp, _target = ep.queries[0]
    _comp, P = settle_samples(net, cfg, start, disp, S=12, n_settle=30, T=cfg.T_floor)
    # samples are not all identical -> the noise floor actually moves the reconstruction
    assert float(np.max(np.std(P, axis=0))) > 0.0
    # and at least somewhere across the query set the argmax actually disagrees (real spread)
    recs = per_query_records(ep, net, cfg, S=12, n_settle=30, T=cfg.T_floor)
    disagreements = [d for (_c, d, _e) in recs]
    assert all(0.0 <= d <= 1.0 for d in disagreements)
    assert max(disagreements) > 0.0          # at least one genuinely ambiguous query


def test_zero_temperature_collapses_spread():
    # With the Langevin noise disabled (SeededRNG.enabled-style T=0), settles are deterministic:
    # disagreement should vanish. This pins down that the spread comes from Pillar-4 noise, not
    # from some other nondeterminism.
    ep = make_episode(h=4, w=4, vocab=5, K=10, seed=3)
    cfg = _small_cfg(seed=3)
    net = train_episode(ep, cfg)
    start, disp, _target = ep.queries[0]
    _comp, P = settle_samples(net, cfg, start, disp, S=8, n_settle=20, T=0.0)
    # T=0 -> identical deterministic trajectories -> zero spread (up to float dust)
    assert float(np.max(np.std(P, axis=0))) < 1e-12


def test_calibration_summary_is_wellformed():
    ep = make_episode(h=4, w=4, vocab=5, K=12, seed=4)
    cfg = _small_cfg(seed=4)
    net = train_episode(ep, cfg)
    d = calibration_for_seed(ep, net, cfg, S=10, n_settle=20, T=cfg.T_floor)
    assert d["n"] > 0
    assert 0.0 <= d["acc"] <= 1.0
    # gap and aurocs are finite-or-nan but never out of band when finite
    if np.isfinite(d["auroc_disagree"]):
        assert 0.0 <= d["auroc_disagree"] <= 1.0
    if np.isfinite(d["gap"]):
        assert -1.0 <= d["gap"] <= 1.0


def test_auroc_endpoints():
    # perfect separation -> 1.0 ; reversed -> 0.0 ; tie -> 0.5
    score = np.array([0.9, 0.8, 0.1, 0.2])
    is_pos = np.array([True, True, False, False])
    assert abs(auroc(score, is_pos) - 1.0) < 1e-9      # tolerance: float AUROC sums aren't bit-exact
    assert abs(auroc(-score, is_pos) - 0.0) < 1e-9
    assert abs(auroc(np.ones(4), is_pos) - 0.5) < 1e-9
    # one class empty -> nan
    assert np.isnan(auroc(score, np.array([False, False, False, False])))
