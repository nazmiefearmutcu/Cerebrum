import numpy as np
from benchmarks.tasks.continual import run_continual


def test_fuse_reduces_forgetting_vs_always_plastic():
    fused   = run_continual(use_fuse=True,  seed=0)
    plastic = run_continual(use_fuse=False, seed=0)
    # both learn A well initially
    assert fused["errA_afterA"] < 1.0 and plastic["errA_afterA"] < 1.0
    # the metaplastic fuse forgets A LESS than always-plastic local learning
    assert fused["forgetA"] < plastic["forgetA"]
    # ...while still learning C (not frozen solid / plastic-death)
    assert fused["errC_afterC"] < fused["errC_beforeC"]


def test_ewc_baseline_runs_and_reduces_forgetting():
    from benchmarks.baselines.ewc import run_continual_ewc
    from benchmarks.tasks.continual import run_continual
    ewc     = run_continual_ewc(seed=0)
    plastic = run_continual(use_fuse=False, seed=0)
    assert ewc["forgetA"] < plastic["forgetA"]    # EWC also reduces forgetting (sanity: the task is learnable-retainable)
    assert ewc["used_fisher_pass"] is True         # EWC requires the extra importance pass GRAIL avoids
