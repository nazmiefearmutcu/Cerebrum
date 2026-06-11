"""Cross-seed robustness of the Stage-3 catastrophic-forgetting result (I4b-ForgetRobust).

The Stage-3 fuse must beat always-plastic learning ROBUSTLY across seeds with a SINGLE
FIXED knob set (no per-task / per-seed retuning of tau_c, alpha_c, beta_c, g_theta). These
tests pin both the strong claim (every-seed-lower AND 95% CIs separated) and the per-seed
guarantee, so a regression in the fuse OR a regression in the measurement (re-introducing
the stochastic settling floor into the eval) trips a test.

NO Fisher pass, NO anchors, NO task-boundary signal enter the fuse — the win is the bare
surprise-gated theta plus a noise-free measurement readout."""
import numpy as np
from benchmarks.tasks.continual import run_continual
from benchmarks.stats import mean_ci

SEEDS = tuple(range(8))


def _forgetA(use_fuse):
    # SINGLE fixed knob set: every seed uses the same continual.py config; we never retune.
    return [run_continual(use_fuse=use_fuse, seed=s)["forgetA"] for s in SEEDS]


def test_eval_is_noise_free_deterministic():
    """The measurement readout must be deterministic: repeated runs of the same seed give
    bit-identical forgetA (no stochastic settling floor leaking into the eval)."""
    a = run_continual(use_fuse=True, seed=1)["forgetA"]
    b = run_continual(use_fuse=True, seed=1)["forgetA"]
    assert a == b


def test_fuse_lower_every_seed():
    """Per-seed guarantee: CEREBRUM-fuse forgetA < always-plastic forgetA on EVERY seed."""
    fuse = _forgetA(True)
    plastic = _forgetA(False)
    for s, (f, p) in zip(SEEDS, zip(fuse, plastic)):
        assert f < p, f"seed {s}: fuse forgetA {f:.3f} !< plastic {p:.3f}"


def test_fuse_ci_separated_from_always_plastic():
    """Strong robustness claim: the 95% CIs do NOT overlap — fuse upper bound is strictly
    below always-plastic lower bound over >=8 seeds with a single fixed knob set."""
    mf, hf = mean_ci(_forgetA(True))
    mp, hp = mean_ci(_forgetA(False))
    assert (mf + hf) < (mp - hp), (
        f"CIs overlap: fuse {mf:.3f}+/-{hf:.3f} (upper {mf+hf:.3f}) vs "
        f"plastic {mp:.3f}+/-{hp:.3f} (lower {mp-hp:.3f})")


def test_fuse_still_learns_C_across_seeds():
    """No plastic-death: the fuse must still learn C (errC_afterC < errC_beforeC) on average,
    so the forgetting win is not bought by freezing solid."""
    res = [run_continual(use_fuse=True, seed=s) for s in SEEDS]
    before = np.mean([r["errC_beforeC"] for r in res])
    after = np.mean([r["errC_afterC"] for r in res])
    assert after < before
