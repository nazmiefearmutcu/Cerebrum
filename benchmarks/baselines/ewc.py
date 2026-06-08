import numpy as np
from grail.pc_core import PCAreas
from grail.plasticity import Eligibility, weight_update, precision_update, feedback_update
from grail.neuromod import Neuromodulator
from grail.rng import SeededRNG
from benchmarks.tasks.continual import _prototypes, _err_on, _make_cfg

# EWC-analog comparator on the SAME local substrate as the GRAIL fuse. The fuse's win is
# matching this WITHOUT EWC's two extra costs: (1) a Fisher-importance pass over A, and
# (2) stored anchor weights W*. Same harness config knobs (_make_cfg) so the comparison is
# apples-to-apples; the only difference is the consolidation mechanism (anchor+penalty vs
# surprise-gated theta).


def run_continual_ewc(seed=0, dim=10, per_task=6, passes=100, lam=5.0):
    cfg = _make_cfg(seed, dim)
    A, B, C = _prototypes(np.random.default_rng(seed+5), 3, per_task, dim)
    net = PCAreas(cfg); nm = Neuromodulator(cfg); rng = SeededRNG(seed)
    elig = [Eligibility((cfg.dims[l+1],), cfg) for l in range(net.L-1)]
    Wstar = [None]*(net.L-1); Omega = [np.zeros_like(net.W[l]) for l in range(net.L-1)]

    def train(patterns, anchored):
        for _ in range(passes):
            for p in patterns:
                for _ in range(cfg.n_settle):
                    net.settle_step(rng, T=cfg.T_floor, clamp_bottom=p)
                    for l in range(net.L-1):
                        elig[l].step(a_pre=net.x[l+1])
                net.compute_errors(); M = nm.update(reward=1.0)
                for l in range(net.L-1):
                    dW = weight_update(M=M, theta=np.ones_like(net.W[l]), Pi_post=net.Pi[l],
                                       eps_post=net.eps[l], elig=elig[l].value, eta=cfg.eta_w/cfg.tau_w)
                    if anchored and Wstar[l] is not None:
                        dW = dW - (cfg.eta_w/cfg.tau_w)*lam*Omega[l]*(net.W[l]-Wstar[l])  # EWC quadratic penalty
                    net.W[l] += dW
                    net.B[l] += (1.0/cfg.tau_b)*feedback_update(net.B[l], a_up=net.x[l+1], eps=net.eps[l], cfg=cfg)
                    net.Pi[l] = precision_update(net.Pi[l], eps_sq=net.eps[l]**2, cfg=cfg)

    train(A, anchored=False)
    errA_afterA = _err_on(net, A, cfg, rng); errC_beforeC = _err_on(net, C, cfg, rng)
    # EWC's extra cost: a Fisher-importance pass over A + stored anchors (GRAIL's fuse needs neither)
    for p in A:
        for _ in range(cfg.n_settle):
            net.settle_step(rng, T=cfg.T_floor, clamp_bottom=p)
            for l in range(net.L-1):
                elig[l].step(a_pre=net.x[l+1])
        net.compute_errors()
        for l in range(net.L-1):
            g = weight_update(M=1.0, theta=np.ones_like(net.W[l]), Pi_post=net.Pi[l],
                              eps_post=net.eps[l], elig=elig[l].value, eta=1.0)
            Omega[l] += g**2
    for l in range(net.L-1):
        Wstar[l] = net.W[l].copy()
    train(B, anchored=True); train(C, anchored=True)
    errA_afterC = _err_on(net, A, cfg, rng); errC_afterC = _err_on(net, C, cfg, rng)
    return {"errA_afterA": errA_afterA, "errA_afterC": errA_afterC, "forgetA": errA_afterC-errA_afterA,
            "errC_beforeC": errC_beforeC, "errC_afterC": errC_afterC, "used_fisher_pass": True}
