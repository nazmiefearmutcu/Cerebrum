"""I5-Unified: integration test for cerebrum/unified.CerebrumNet — the ONE network that exercises
all five pillars together (PC areas + grid generative prior + basal-ganglia gate + k<<n
workspace broadcast + surprise-gated metaplastic fuse on module weights), all driven by the
single scalar neuromodulator M.

These tests pin the load-bearing invariants end-to-end (NOT just that it runs):
  - the workspace write is strict one-hot (BAN-1 via assert_one_hot),
  - the neuromodulator M is a scalar (BAN-2 via assert_scalar_M / ndim==0),
  - the grid driver is Exogenous-only (BAN-5: a data-dependent action is a type error),
  - the metaplastic theta ACTUALLY gates module plasticity (a closed fuse shrinks |dW|),
  - a basic learning signal works (reconstruction error drops on a repeated pattern),
  - counters increment (learn-time scalar-M O(1) + infer-time broadcast vectors).
"""
import numpy as np
import pytest

from cerebrum.config import CerebrumConfig
from cerebrum.unified import CerebrumNet
from cerebrum.types import Exogenous
from cerebrum.invariants import assert_one_hot, assert_scalar_M


def _make_net(seed=0, n_modules=3, k_slots=2, slice_dim=4):
    cfg = CerebrumConfig(dims=(slice_dim, 8), n_settle=8, seed=seed)
    return CerebrumNet(n_modules=n_modules, k_slots=k_slots, slice_dim=slice_dim, cfg=cfg), cfg


def _obs(n_modules, slice_dim, rng):
    return [rng.standard_normal(slice_dim) * 0.4 for _ in range(n_modules)]


def test_step_runs_end_to_end_and_returns_scalar_M():
    net, _ = _make_net()
    rng = np.random.default_rng(0)
    obs = _obs(3, 4, rng)
    action = Exogenous(np.array([0.3, -0.1]))
    z, M = net.step(obs, action=action, reward=1.0)
    assert z.shape == (3, 2)
    assert np.ndim(M) == 0            # scalar neuromodulator (BAN-2)
    assert_scalar_M(M)


def test_workspace_write_is_one_hot():
    net, _ = _make_net()
    rng = np.random.default_rng(1)
    for _ in range(5):
        z, _ = net.step(_obs(3, 4, rng), action=Exogenous(np.array([0.1, 0.2])), reward=1.0)
        assert_one_hot(z, axis=0)     # strict one-hot routing (BAN-1)


def test_grid_driver_must_be_exogenous():
    """A raw (data-dependent) action is a BAN-5 type error — only Exogenous can drive the grid."""
    net, _ = _make_net()
    rng = np.random.default_rng(2)
    with pytest.raises(TypeError):
        net.step(_obs(3, 4, rng), action=np.array([0.1, 0.2]), reward=1.0)


def test_counters_increment():
    net, _ = _make_net()
    rng = np.random.default_rng(3)
    net.step(_obs(3, 4, rng), action=Exogenous(np.array([0.1, 0.0])), reward=1.0)
    assert net.counters.global_comm_learn >= 1      # O(1) scalar-M learn-time event
    assert net.counters.global_comm_infer > 0       # broadcast vectors at infer time
    assert net.counters.synaptic_ops > 0


def test_metaplastic_theta_gates_module_plasticity():
    """The per-synapse theta from MetaplasticFuse must actually multiply the four-factor module
    weight update: with the fuse forced CLOSED (theta->0) the weight change is strictly smaller
    in magnitude than with the fuse forced OPEN (theta=1), on the SAME inputs/RNG."""
    rng = np.random.default_rng(7)
    action = Exogenous(np.array([0.2, 0.1]))
    obs = _obs(3, 4, rng)

    def total_dW(force_theta):
        net, _ = _make_net(seed=5)
        net._force_theta = force_theta            # test hook: pin theta to a constant
        W_before = [m.W[0].copy() for m in net.modules]
        net.step([o.copy() for o in obs], action=action, reward=1.0)
        return sum(float(np.sum(np.abs(np.asarray(m.W[0]) - np.asarray(wb)))) for m, wb in zip(net.modules, W_before))

    open_change = total_dW(1.0)     # fuse OPEN: full plasticity
    closed_change = total_dW(0.0)   # fuse CLOSED: theta=0 should zero the module weight update
    assert open_change > 0.0
    assert closed_change < open_change
    assert closed_change == pytest.approx(0.0, abs=1e-12)


def test_fuse_object_is_used_and_theta_in_unit_interval():
    """The net owns a MetaplasticFuse per module/layer and the theta it produces is in [0,1]."""
    net, _ = _make_net()
    rng = np.random.default_rng(11)
    net.step(_obs(3, 4, rng), action=Exogenous(np.array([0.0, 0.1])), reward=1.0)
    assert len(net.fuse) == net.M_
    th = np.asarray(net.last_theta[0][0])
    assert np.all(th >= 0.0) and np.all(th <= 1.0)


def test_basic_learning_signal_error_drops():
    """Pillar-2 sanity: repeatedly presenting the SAME observation stream with positive reward
    drives the module BOTTOM-AREA reconstruction error down (the local four-factor rule learns).

    The grid binding rate is set to 0 here so the structural top-down prediction is a fixed
    (neutral) target and the probe isolates the module's own generative weights `W` — the grid
    path-integration is verified separately in test_grid_path_integration_advances_top_down. The
    harness uses a visible learning rate (eta_w/tau_w large, the same regime as the continual
    benchmark) and a deterministic noise-free (T=0) readout with a neutral broadcast, so eps[0]
    reflects the learned WEIGHTS rather than the settling floor or the workspace recurrence."""
    cfg = CerebrumConfig(dims=(4, 8), n_settle=10, seed=3, grid_eta_bind=0.0,
                      eta_w=0.6, tau_w=1.0, tau_e=1.0, tau_r=1e9)
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg)
    rng = np.random.default_rng(0)
    obs = [rng.standard_normal(4) * 0.5 for _ in range(2)]
    still = Exogenous(np.array([0.0, 0.0]))

    def bottom_err():
        net.workspace.slots[:] = 0.0                      # neutral broadcast for the readout
        net.settle_only(obs, action=still, T=0.0)         # noise-free: reflects the weights
        return float(np.mean([float(np.sum(np.asarray(m.eps[0]) ** 2)) for m in net.modules]))

    e0 = bottom_err()
    for _ in range(60):
        net.step([o.copy() for o in obs], action=still, reward=1.0)
    e1 = bottom_err()
    assert e1 < e0, f"error did not drop: {e0:.4f} -> {e1:.4f}"


def test_grid_path_integration_advances_top_down():
    """Pillar-3: an Exogenous action path-integrates the grid head, so the structural top-down
    prediction changes after a move (the grid prior is actually in the loop)."""
    net, _ = _make_net()
    rng = np.random.default_rng(0)
    obs = _obs(3, 4, rng)
    net.step(obs, action=Exogenous(np.array([0.0, 0.0])), reward=1.0)
    td0 = net.last_top_pred.detach().cpu().numpy()
    net.step(obs, action=Exogenous(np.array([1.0, 0.5])), reward=1.0)
    td1 = net.last_top_pred.detach().cpu().numpy()
    assert not np.allclose(td0, td1)
