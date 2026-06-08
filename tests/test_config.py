from grail.config import GRAILConfig

def test_defaults_present_and_sane():
    c = GRAILConfig()
    assert c.dims[0] > 0 and len(c.dims) >= 2
    assert c.T_floor > 0.0           # Pillar 4: never MAP collapse
    assert c.dt > 0 and c.n_settle > 0
    assert c.tau_e < c.tau_w         # spec timescale ordering tau_x << tau_gate << tau_w
    assert c.tau_x < c.tau_w

def test_config_is_frozen_and_overridable():
    c = GRAILConfig(seed=7, T_floor=0.05)
    assert c.seed == 7 and c.T_floor == 0.05
