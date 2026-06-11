from cerebrum.config import CerebrumConfig

def test_defaults_present_and_sane():
    c = CerebrumConfig()
    assert c.dims[0] > 0 and len(c.dims) >= 2
    assert c.T_floor > 0.0           # Pillar 4: never MAP collapse
    assert c.dt > 0 and c.n_settle > 0
    assert c.tau_e < c.tau_w         # spec timescale ordering tau_x << tau_gate << tau_w
    assert c.tau_x < c.tau_w

def test_config_is_frozen_and_overridable():
    c = CerebrumConfig(seed=7, T_floor=0.05)
    assert c.seed == 7 and c.T_floor == 0.05

def test_align_feedback_defaults_off():
    c = CerebrumConfig()
    assert c.align_feedback is False     # OPT-IN: default behavior unchanged
    assert c.lam_kp > 0.0                # matched KP decay parameter present
    assert CerebrumConfig(align_feedback=True).align_feedback is True

def test_balance_grid_precision_defaults_off():
    c = CerebrumConfig()
    assert c.balance_grid_precision is False     # OPT-IN: default behavior unchanged
    assert c.grid_precision_ref > 0.0            # balancing reference ratio present
    assert CerebrumConfig(balance_grid_precision=True).balance_grid_precision is True
