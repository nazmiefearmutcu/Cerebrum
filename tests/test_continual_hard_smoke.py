"""Fast smoke test for the HARD catastrophic-forgetting probe (FM4 stress-test).

The hard probe pushes the surprise-gated metaplastic fuse to two stress axes with a SINGLE
FIXED knob set (no per-task retuning, no Fisher pass, no anchors, no task-boundary signal —
those would be BAN violations; they live only in ewc.py):

  (1) LONGER streams  : 8-10 sequential tasks A..J, tracking first-task (A) protection.
  (2) TASK SIMILARITY : tasks that share an input subspace (overlapping prototypes), swept by
                        a single similarity knob in [0,1].

These asserts only pin robustly-true, finite-output invariants — NOT the (knife-edge) win,
which is the empirical question the probe answers. The probe is allowed to find a BREAK."""
import numpy as np
from benchmarks.tasks.continual import run_continual_stream


def test_stream_runs_finite_and_tracks_first_task():
    """An N-task stream returns finite first-task numbers and a per-task forgetting vector."""
    r = run_continual_stream(use_fuse=True, seed=0, n_tasks=4, similarity=0.0, passes=40)
    assert np.isfinite(r["errA_afterA"])         # A is learned at all
    assert np.isfinite(r["errA_final"])          # A measured after the whole stream
    assert r["errA_afterA"] >= 0.0
    # forgetA == errA_final - errA_afterA, exactly (definition, not a claim about its sign)
    assert abs(r["forgetA"] - (r["errA_final"] - r["errA_afterA"])) < 1e-9
    # per-task forgetting curve has one entry per task that has a measurable drift (>=1)
    assert len(r["forget_curve"]) == 4
    assert all(np.isfinite(x) for x in r["forget_curve"])
    # the fuse reports a finite mean consolidation reserve in [0, c_max]
    assert 0.0 <= r["cbar"] <= 1.0


def test_similarity_knob_changes_the_stream():
    """The similarity knob must actually change the prototypes (s=0 independent vs s=1 shared
    subspace), so sweeping it is meaningful. We assert the two extremes differ, not a winner."""
    lo = run_continual_stream(use_fuse=True, seed=1, n_tasks=3, similarity=0.0, passes=40)
    hi = run_continual_stream(use_fuse=True, seed=1, n_tasks=3, similarity=1.0, passes=40)
    assert np.isfinite(lo["forgetA"]) and np.isfinite(hi["forgetA"])
    # at s=1 every task shares the same subspace anchor -> the streams are genuinely different
    assert lo["errA_afterA"] != hi["errA_afterA"] or lo["forgetA"] != hi["forgetA"]


def test_plastic_baseline_runs_on_stream():
    """always-plastic (theta==1) must also run on the generalized stream so the probe can
    compare fuse vs always-plastic at every step (no fuse-only path)."""
    p = run_continual_stream(use_fuse=False, seed=0, n_tasks=4, similarity=0.5, passes=40)
    assert np.isfinite(p["forgetA"])
    assert p["cbar"] == 0.0     # no fuse -> no consolidation reserve reported
    assert len(p["forget_curve"]) == 4


def test_determinism_of_noise_free_eval():
    """The T=0 measurement readout is deterministic: same args -> bit-identical forgetA."""
    a = run_continual_stream(use_fuse=True, seed=2, n_tasks=5, similarity=0.3, passes=30)["forgetA"]
    b = run_continual_stream(use_fuse=True, seed=2, n_tasks=5, similarity=0.3, passes=30)["forgetA"]
    assert a == b
