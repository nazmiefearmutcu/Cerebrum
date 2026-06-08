import numpy as np
from grail.config import GRAILConfig
from grail.pc_core import PCAreas
from grail.rng import SeededRNG

def test_deterministic_settling_reduces_energy():
    c = GRAILConfig(dims=(6,5,4), seed=1, T_floor=0.0, n_settle=80, dt=0.05, gamma=0.0)
    pc = PCAreas(c)
    # symmetric limit for a clean Lyapunov check: set B[l] = W[l].T
    for l in range(pc.L-1): pc.B[l] = pc.W[l].T.copy()
    obs = np.array([0.3,-0.2,0.5,0.1,-0.4,0.2])
    pc.x[0][:] = obs
    pc.compute_errors()
    e0 = pc.energy()
    rng = SeededRNG(0, enabled=False)  # T=0 + no noise => deterministic descent
    for _ in range(c.n_settle):
        pc.settle_step(rng, T=0.0, clamp_bottom=obs)
    pc.compute_errors()
    assert pc.energy() < e0           # deterministic settling descends the surrogate energy

def test_noise_prevents_collapse():
    c = GRAILConfig(dims=(5,4), seed=2, T_floor=0.1, dt=0.05)
    pc = PCAreas(c); obs = np.array([0.2,0.1,-0.3,0.4,0.0])
    pc.x[0][:] = obs; pc.compute_errors()
    rng = SeededRNG(3, enabled=True)
    xs = []
    for _ in range(200):
        pc.settle_step(rng, T=c.T_floor, clamp_bottom=obs); xs.append(pc.x[1].copy())
    var = np.var(np.array(xs[50:]), axis=0)
    assert np.all(var > 0)            # Pillar 4: stays a sampler, never a fixed point
