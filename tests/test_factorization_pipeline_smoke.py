"""Smoke test for the FULL-PIPELINE factorization probe (benchmarks/run_factorization_pipeline.py).

CONTEXT — C3-FullPipeline
-------------------------
The 0.92 held-out factor-decode finding (benchmarks/run_factorization.py) was measured on a BARE
PCAreas hierarchy trained by the local four-factor rule. This probe asks: does that factored,
compositionally-generalizing latent SURVIVE when the SAME cortical module operates inside the
richer unified dynamics — with the grid-HEAD structural top-down prediction active, and/or the
thalamo-cortical workspace broadcast feeding back, and/or the surprise-gated metaplastic fuse
gating the local plasticity? We linear-probe each factor off the module's latent on HELD-OUT
combos under each condition (bare / +grid / +broadcast / +fuse / full-CerebrumNet), with the same
UNTRAINED and RANDOM-PROJECTION controls as the bare probe.

These assertions only check ROBUSTLY-TRUE things (finite/in-range outputs, shape/coverage
contracts, the noise-free measurement is deterministic, each pipeline condition is genuinely
DIFFERENT from bare in its dynamics, the CerebrumNet path runs and preserves its invariants). They
do NOT assert "factorization survives" — that is the empirical question the run script answers.
"""
import numpy as np

from cerebrum.config import CerebrumConfig
from cerebrum.pc_core import PCAreas
from benchmarks.tasks.compositional import CompositionalTask
from benchmarks.run_factorization import make_split
from benchmarks.run_factorization_pipeline import (
    PipelineConfig, train_pipeline_module, settle_top_latent_pipeline,
    pipeline_probe, run_one_seed_pipeline, CONDITIONS,
)


def test_conditions_cover_the_advertised_axes():
    # the probe must compare bare vs each added pipeline piece vs the full CerebrumNet.
    names = set(CONDITIONS)
    assert {"bare", "grid", "broadcast", "fuse", "full"}.issubset(names)


def test_bare_pipeline_matches_train_pc_latent_bitwise():
    # the "bare" pipeline condition (no grid/broadcast/fuse) must reproduce the SAME local rule
    # as the original probe's _train_pc, so the baseline is a true apples-to-apples reference and
    # any drop under a richer condition is attributable to the ADDED dynamics, not a reimplementation.
    from benchmarks.tasks.compositional import _train_pc
    from benchmarks.run_factorization import settle_top_latent as stl
    task = CompositionalTask(A=4, B=4, part_dim=6, seed=0)
    dims = (task.obs_dim, 12, 12)
    cfg = CerebrumConfig(dims=dims, n_settle=10, seed=0)
    ref = _train_pc(task, cfg, passes=8)
    pc = PipelineConfig(condition="bare")
    got = train_pipeline_module(task, cfg, pc, passes=8)
    # same learned forward weights and feedback weights
    for l in range(ref.L - 1):
        assert np.allclose(ref.W[l], got.W[l], atol=1e-10), f"W[{l}] differs from _train_pc"
        assert np.allclose(ref.B[l], got.B[l], atol=1e-10), f"B[{l}] differs from _train_pc"
    # and the noise-free read-out matches the original probe's settle_top_latent
    obs = task.embed(1, 2)
    z_ref = stl(ref, obs, steps=20)
    z_got = settle_top_latent_pipeline(got, obs, steps=20, pcfg=pc)
    assert np.allclose(z_ref, z_got, atol=1e-10)


def test_settle_latent_is_finite_and_deterministic_each_condition():
    task = CompositionalTask(A=4, B=4, part_dim=6, seed=1)
    dims = (task.obs_dim, 12, 12)
    cfg = CerebrumConfig(dims=dims, n_settle=8, seed=1)
    obs = task.embed(0, 3)
    for cond in ("bare", "grid", "broadcast", "fuse"):
        pc = PipelineConfig(condition=cond)
        net = train_pipeline_module(task, cfg, pc, passes=4)
        z1 = settle_top_latent_pipeline(net, obs, steps=12, pcfg=pc)
        z2 = settle_top_latent_pipeline(net, obs, steps=12, pcfg=pc)
        assert z1.shape == (dims[-1],)
        assert np.all(np.isfinite(z1))
        assert np.array_equal(z1, z2), f"noise-free readout not deterministic for {cond}"


def test_grid_condition_actually_injects_a_nonzero_topdown():
    # the +grid condition must put a genuinely NON-trivial structural top-down prediction into the
    # module settle (otherwise it would be silently identical to bare).
    task = CompositionalTask(A=4, B=4, part_dim=6, seed=2)
    dims = (task.obs_dim, 12, 12)
    cfg = CerebrumConfig(dims=dims, n_settle=6, seed=2)
    pc = PipelineConfig(condition="grid")
    net = train_pipeline_module(task, cfg, pc, passes=4)
    # after training the grid content store must be bound (non-zero) so its top_pred is live
    assert pc._grid is not None and pc._grid.store is not None
    assert np.linalg.norm(pc._grid.complete()) > 0.0


def test_broadcast_condition_feeds_back_nonzero_workspace():
    task = CompositionalTask(A=4, B=4, part_dim=6, seed=3)
    dims = (task.obs_dim, 12, 12)
    cfg = CerebrumConfig(dims=dims, n_settle=6, seed=3)
    pc = PipelineConfig(condition="broadcast")
    net = train_pipeline_module(task, cfg, pc, passes=6)
    # broadcast slot must be populated (the module wrote its own read into the 1-slot workspace)
    assert pc._wksp is not None
    assert np.linalg.norm(pc._wksp.broadcast()) > 0.0


def test_fuse_condition_gates_plasticity_smaller_dw_than_bare():
    # the +fuse condition multiplies the four-factor update by theta in [0,1], so over training the
    # NET weight movement must be no larger than the un-gated bare condition (theta<=1 shrinks dW).
    task = CompositionalTask(A=4, B=4, part_dim=6, seed=4)
    dims = (task.obs_dim, 12, 12)
    cfg = CerebrumConfig(dims=dims, n_settle=8, seed=4)
    bare = train_pipeline_module(task, cfg, PipelineConfig(condition="bare"), passes=10)
    fused = train_pipeline_module(task, cfg, PipelineConfig(condition="fuse"), passes=10)
    init = PCAreas(cfg)
    mv_bare = sum(float(np.sum(np.abs(np.asarray(bare.W[l]) - np.asarray(init.W[l])))) for l in range(bare.L - 1))
    mv_fuse = sum(float(np.sum(np.abs(np.asarray(fused.W[l]) - np.asarray(init.W[l])))) for l in range(fused.L - 1))
    assert mv_fuse <= mv_bare + 1e-9, f"fuse did not gate plasticity: {mv_fuse} > {mv_bare}"
    assert mv_fuse >= 0.0


def test_full_cerebrumnet_condition_runs_and_returns_in_range_decode():
    # the full-CerebrumNet path (n_modules=1, real gate/workspace/grid/fuse) must produce a finite,
    # in-range held-out decode and a finite untrained/random-projection control.
    task = CompositionalTask(A=4, B=4, part_dim=6, seed=0)
    train, held = make_split(A=4, B=4, frac_heldout=0.3, seed=10)
    res = pipeline_probe(task, train, held, dims=(task.obs_dim, 12, 12),
                         condition="full", passes=6, seed=0, decoder="ncm")
    for k in ("trained_f1", "trained_f2", "untrained_f1", "untrained_f2",
              "randproj_f1", "randproj_f2"):
        assert k in res
        assert np.isfinite(res[k]) and 0.0 <= res[k] <= 1.0


def test_pipeline_probe_in_range_all_conditions():
    task = CompositionalTask(A=4, B=4, part_dim=6, seed=1)
    train, held = make_split(A=4, B=4, frac_heldout=0.3, seed=11)
    for cond in CONDITIONS:
        res = pipeline_probe(task, train, held, dims=(task.obs_dim, 12, 12),
                             condition=cond, passes=6, seed=0, decoder="ncm")
        for k in ("trained_f1", "trained_f2", "untrained_f1", "untrained_f2",
                  "randproj_f1", "randproj_f2"):
            assert np.isfinite(res[k]) and 0.0 <= res[k] <= 1.0


def test_grid_blows_up_latent_norm_vs_bare_mechanism():
    # MECHANISM check (robustly true): the +grid structural top-down prediction comes from the
    # grid HEAD's (never-decayed) Hebbian content store, which DOMINATES the small obs-driven
    # cortical latent. So the trained-latent L2 norm under +grid must be MUCH larger than bare.
    # This is the signature behind the grid/full decode collapse, and it is a stable, large effect.
    task = CompositionalTask(A=4, B=4, part_dim=6, seed=0)
    train, held = make_split(A=4, B=4, frac_heldout=0.3, seed=10)
    dims = (task.obs_dim, 12, 12)
    bare = pipeline_probe(task, train, held, dims=dims, condition="bare",
                          passes=30, seed=0, decoder="ncm")
    grid = pipeline_probe(task, train, held, dims=dims, condition="grid",
                          passes=30, seed=0, decoder="ncm")
    assert np.isfinite(bare["latent_norm"]) and np.isfinite(grid["latent_norm"])
    assert grid["latent_norm"] > 5.0 * bare["latent_norm"], (
        f"grid latent norm {grid['latent_norm']:.3f} not >> bare {bare['latent_norm']:.3f}")


def test_run_one_seed_pipeline_has_all_conditions_and_both_probes():
    out = run_one_seed_pipeline(seed=0, A=4, B=4, part_dim=6, width=12, depth=3,
                                frac_heldout=0.3, passes=6)
    for cond in CONDITIONS:
        assert cond in out
        for kind in ("ncm", "logreg"):
            assert kind in out[cond]
            for k in ("trained_f1", "trained_f2", "untrained_f1", "untrained_f2",
                      "randproj_f1", "randproj_f2"):
                assert np.isfinite(out[cond][kind][k]) and 0.0 <= out[cond][kind][k] <= 1.0
