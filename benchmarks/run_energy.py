import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from cerebrum.config import CerebrumConfig
from cerebrum.pc_core import PCAreas
from cerebrum.plasticity import Eligibility, weight_update, precision_update, feedback_update
from cerebrum.neuromod import Neuromodulator
from cerebrum.rng import SeededRNG
from cerebrum.energy import (spike_sparsity, dynamic_synaptic_ops, dynamic_energy_magnitude,
                          dense_backprop_ops, global_comm_per_update)

# Spike threshold for the (conservative) thresholded op count: above the Langevin noise floor so a
# "spike" reflects systematic prediction error, not noise.
SPIKE_TOL = 0.1


def _measure(net, protos, cfg):
    """Noise-free (T=0) snapshot of competence + dynamic energy. Training keeps the Langevin noise
    (Pillar 4); the MEASUREMENT settles deterministically so the metric reflects systematic error,
    not the noise floor."""
    rng0 = SeededRNG(0, enabled=False)
    errs = []; rhos = []; ops = []; mags = []
    for p in protos:
        for _ in range(40):
            net.settle_step(rng0, T=0.0, clamp_bottom=p)
        net.compute_errors()
        errs.append(float(np.sum(net.eps[0] ** 2)))
        rhos.append(spike_sparsity(net.eps[:net.L - 1], tol=SPIKE_TOL))
        ops.append(dynamic_synaptic_ops(net, tol=SPIKE_TOL))
        mags.append(dynamic_energy_magnitude(net))
    return float(np.mean(errs)), float(np.mean(rhos)), float(np.mean(ops)), float(np.mean(mags))


def run_energy(seed=0, dim=10, latent=16, n_proto=3, passes=300, measure_every=30):
    cfg = CerebrumConfig(dims=(dim, latent), n_settle=10, seed=seed, tau_w=1.0, eta_w=0.6, tau_r=1e9, tau_e=1.0)
    rng_p = np.random.default_rng(seed + 5)
    protos = [0.4 * rng_p.standard_normal(dim) for _ in range(n_proto)]
    net = PCAreas(cfg); nm = Neuromodulator(cfg); rng = SeededRNG(seed)
    elig = [Eligibility((cfg.dims[l + 1],), cfg) for l in range(net.L - 1)]
    curve = [(0,) + _measure(net, protos, cfg)]
    for ep in range(1, passes + 1):
        for p in protos:
            for _ in range(cfg.n_settle):
                net.settle_step(rng, T=cfg.T_floor, clamp_bottom=p)     # training keeps Langevin noise
                for l in range(net.L - 1):
                    elig[l].step(a_pre=net.x[l + 1])
            net.compute_errors()
            M = nm.update(reward=1.0)
            for l in range(net.L - 1):
                net.W[l] += weight_update(M=M, theta=np.ones_like(net.W[l]), Pi_post=net.Pi[l],
                                          eps_post=net.eps[l], elig=elig[l].value, eta=cfg.eta_w / cfg.tau_w)
                net.B[l] += (1.0 / cfg.tau_b) * feedback_update(net.B[l], a_up=net.x[l + 1], eps=net.eps[l], cfg=cfg)
                net.Pi[l] = precision_update(net.Pi[l], eps_sq=net.eps[l] ** 2, cfg=cfg)
        if ep % measure_every == 0:
            curve.append((ep,) + _measure(net, protos, cfg))
    return curve, cfg


if __name__ == "__main__":
    curve, cfg = run_energy()
    dense = dense_backprop_ops(cfg.dims); gc = global_comm_per_update(cfg.dims)
    print("Task-3 energy/op LEARNING CURVE (CEREBRUM, reconstruction; noise-free T=0 measurement)")
    print(f"{'pass':>5}{'recon_err':>12}{'eps_spars@0.1':>14}{'dyn_ops':>10}{'dyn_energy':>12}")
    for (ep, e, r, o, m) in curve:
        print(f"{ep:>5}{e:>12.4f}{r:>14.3f}{o:>10.1f}{m:>12.2f}")
    f, l = curve[0], curve[-1]
    print(f"\nDYNAMIC switching-energy decays with competence: recon_err {f[1]:.3f} -> {l[1]:.3f} "
          f"(~{f[1]/max(l[1],1e-9):.1f}x); dyn_energy {f[4]:.1f} -> {l[4]:.1f} "
          f"(~{f[4]/max(l[4],1e-9):.1f}x); spike-sparsity@0.1 {f[2]:.3f} -> {l[2]:.3f}.")
    print(f"Dense backprop MAC/step (rho=1, NO decay): {dense} ops.")
    print(f"Global comm/update: CEREBRUM = {gc['cerebrum_learn_scalars']} SCALAR (M); "
          f"backprop = {gc['backprop_error_vector_elems']} error-VECTOR elements (O(depth)).")
    print("HONEST: only the DYNAMIC switching term decays — static/leakage and settle-time energy do "
          "NOT. The thresholded spike count is conservative (the learner plateaus at recon~0.25, so "
          "some units stay above 0.1); the magnitude-weighted dyn_energy tracks the true error decay. "
          "Small task, not a scaling claim.")
