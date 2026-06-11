import math
import numpy as np

from benchmarks.run_scaling import (
    probe_task1_scaling,
    probe_forgetting_scaling,
    probe_depth_scaling,
    verdict_task1,
    verdict_forgetting,
)


def _finite(x):
    return isinstance(x, float) and math.isfinite(x)


def test_task1_scaling_produces_finite_numbers():
    # tiny config: one small size, 2 seeds, few MLP epochs -> still must produce finite means + CIs
    res = probe_task1_scaling(sizes=((4, 4, 5),), Ks=(5,), seeds=(0, 1), mlp_epochs=20)
    assert (4, 4, 5) in res
    row = res[(4, 4, 5)][5]
    for method in ("cerebrum", "flat", "mlp"):
        assert _finite(row[method]["mean"])
        assert _finite(row[method]["ci"])
        assert row[method]["ci"] >= 0.0
        assert len(row[method]["raw"]) == 2
        for v in row[method]["raw"]:
            assert 0.0 <= v <= 1.0
    # chance level recorded and finite
    assert _finite(row["chance"])


def test_forgetting_scaling_more_tasks_produces_finite_numbers():
    # A->B->C with the fuse and always-plastic; report forgetting of the FIRST task
    res = probe_forgetting_scaling(n_tasks=3, seeds=(0, 1), passes=20, per_task=4)
    for method in ("fuse", "plastic"):
        assert method in res
        # forget_first[m] = forgetting of task A after m further tasks have been learned
        assert "forget_first" in res[method]
        for m, stat in res[method]["forget_first"].items():
            assert _finite(stat["mean"])
            assert _finite(stat["ci"])
            assert len(stat["raw"]) == 2
        # still learns the LAST task (not plastic-death)
        assert _finite(res[method]["learn_last"]["mean"])


def test_depth_scaling_produces_finite_numbers():
    res = probe_depth_scaling(depths=(2, 3), Ks=(5,), seeds=(0, 1), h=4, w=4, vocab=5)
    for depth in (2, 3):
        assert depth in res
        stat = res[depth][5]["cerebrum"]
        assert _finite(stat["mean"])
        assert _finite(stat["ci"])
        assert len(stat["raw"]) == 2


def test_verdict_helpers_return_strings():
    res = probe_task1_scaling(sizes=((4, 4, 5),), Ks=(5,), seeds=(0, 1), mlp_epochs=20)
    v = verdict_task1(res)
    assert isinstance(v, str) and len(v) > 0
    res2 = probe_forgetting_scaling(n_tasks=3, seeds=(0, 1), passes=20, per_task=4)
    v2 = verdict_forgetting(res2)
    assert isinstance(v2, str) and len(v2) > 0
