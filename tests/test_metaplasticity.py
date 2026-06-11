import numpy as np
from cerebrum.config import CerebrumConfig
from cerebrum.metaplasticity import MetaplasticFuse

def test_metaplasticity_config_present():
    c = CerebrumConfig()
    assert c.c_max > 0
    assert c.tau_c > c.tau_S      # consolidation slower than the surprise baseline EMA
    assert c.alpha_c > 0 and c.beta_c > 0 and c.g_theta > 0

def test_sustained_low_surprise_builds_reserve_and_closes_fuse():
    c = CerebrumConfig(tau_c=5.0, tau_S=2.0)
    fuse = MetaplasticFuse(shape=(2, 2), cfg=c)
    Pi = np.array([1.0, 1.0]); eps = np.array([0.0, 0.0]); elig = np.array([0.0, 0.0])  # perfectly predicted -> low surprise
    th = None
    for _ in range(500): th = fuse.update(Pi, eps, elig)
    assert np.all(np.asarray(fuse.c) > 0.5)            # reserve builds under sustained low surprise
    assert np.all(np.asarray(th) < 0.5)                # fuse closes (consolidated synapses freeze)

def test_high_surprise_opens_fuse():
    c = CerebrumConfig(tau_c=5.0, tau_S=2.0)
    fuse = MetaplasticFuse(shape=(2, 2), cfg=c)
    # first consolidate under low surprise
    for _ in range(500): fuse.update(np.array([1., 1.]), np.array([0., 0.]), np.array([0., 0.]))
    # then a surprising event (large eps*elig) should push theta back up
    th = fuse.update(np.array([1., 1.]), np.array([3.0, 3.0]), np.array([3.0, 3.0]))
    assert np.all(np.asarray(th) > 0.3)                # surprise reopens plasticity (learn-on-surprise)

def test_theta_in_unit_interval():
    fuse = MetaplasticFuse(shape=(3, 4), cfg=CerebrumConfig())
    th = fuse.update(np.ones(3), np.random.default_rng(0).standard_normal(3), np.ones(4))
    assert th.shape == (3, 4) and np.all(np.asarray(th) >= 0) and np.all(np.asarray(th) <= 1)
