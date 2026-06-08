from grail.config import GRAILConfig


def test_metaplasticity_config_present():
    c = GRAILConfig()
    assert c.c_max > 0
    assert c.tau_c > c.tau_S      # consolidation slower than the surprise baseline EMA
    assert c.alpha_c > 0 and c.beta_c > 0 and c.g_theta > 0
