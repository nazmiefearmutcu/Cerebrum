"""Fast smoke test for the large-graph scaling runner (benchmarks/run_largegraph.py).

Uses a tiny config (one small size, K=5 only, 2 seeds, few MLP epochs) so the test
runs in well under a second, but still exercises the SAME code path the full
12x12 / 16x16 run uses. We only assert that the runner produces finite, in-range
numbers + CIs + a chance level + a string verdict — NOT any accuracy threshold
(that would be a result claim, which belongs in the honest report, not a test)."""
import math

from benchmarks.run_largegraph import probe_largegraph, verdict_largegraph


def _finite(x):
    return isinstance(x, float) and math.isfinite(x)


def test_largegraph_produces_finite_numbers():
    # tiny: one small size, one K, 2 seeds, few MLP epochs
    res = probe_largegraph(sizes=((4, 4, 5),), Ks=(5,), seeds=(0, 1),
                           mlp_epochs=20, n_settle=6)
    assert (4, 4, 5) in res
    row = res[(4, 4, 5)][5]
    for method in ("grail", "flat", "mlp"):
        assert _finite(row[method]["mean"])
        assert _finite(row[method]["ci"])
        assert row[method]["ci"] >= 0.0
        assert len(row[method]["raw"]) == 2
        for v in row[method]["raw"]:
            assert 0.0 <= v <= 1.0
    # chance level recorded, finite, equals 1/vocab
    assert _finite(row["chance"])
    assert abs(row["chance"] - 1.0 / 5) < 1e-9
    # coverage fraction recorded and in (0, 1]
    assert _finite(row["coverage"])
    assert 0.0 < row["coverage"] <= 1.0


def test_largegraph_verdict_is_a_string():
    res = probe_largegraph(sizes=((4, 4, 5),), Ks=(5,), seeds=(0, 1),
                           mlp_epochs=20, n_settle=6)
    v = verdict_largegraph(res)
    assert isinstance(v, str) and len(v) > 0
