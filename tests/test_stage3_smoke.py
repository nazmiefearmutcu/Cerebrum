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
