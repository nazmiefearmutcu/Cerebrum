import numpy as np
from grail.config import GRAILConfig
from grail.pc_core import PCAreas
from grail.plasticity import Eligibility, weight_update, precision_update, feedback_update
from grail.metaplasticity import MetaplasticFuse
from grail.neuromod import Neuromodulator
from grail.rng import SeededRNG

# --- Continual-learning harness config knobs (FM4 knife-edge; see concerns / README) ---
# Patterns are scaled into the tanh decoder's representable range (PROTO_SCALE) so the
# reconstruction floor is not dominated by saturation; the latent area (LATENT) gives the
# decoder enough capacity that A is genuinely learnable (errA_afterA < 1) yet the SHARED
# weights are overwritten by B/C (real catastrophic forgetting). tau_r is large so the
# reward-prediction-error M stays positive across the whole stream (constant reward=1).
PROTO_SCALE = 0.4
LATENT = 16
ETA_W = 0.6
TAU_W = 1.0
TAU_E = 1.0
TAU_R = 1e9
# Metaplasticity knobs tuned so A consolidates (cbar high) during its passes BEFORE B/C
# arrive, while B/C are still surprising enough to re-erode c and reopen theta (no
# plastic-death). NEVER a Fisher pass / anchor / task-boundary — pure local surprise.
TAU_C = 80.0
ALPHA_C = 1.0
BETA_C = 4.0
G_THETA = 3.5
TAU_S = 20.0


def _make_cfg(seed, dim):
    return GRAILConfig(
        dims=(dim, LATENT), n_settle=10, seed=seed,
        tau_w=TAU_W, eta_w=ETA_W, tau_e=TAU_E, tau_r=TAU_R,
        tau_c=TAU_C, alpha_c=ALPHA_C, beta_c=BETA_C, g_theta=G_THETA, tau_S=TAU_S,
    )


def _prototypes(rng, n_tasks, per_task, dim):
    return [[PROTO_SCALE * rng.standard_normal(dim) for _ in range(per_task)] for _ in range(n_tasks)]


# Fixed seed for the noise-free MEASUREMENT readout. The training/inference floor T_floor>0
# is a learning-time regularizer (forbids MAP collapse, Pillar 4); but a MEASUREMENT of the
# already-learned weights must not re-inject that stochastic floor, or it adds per-eval
# variance (~0.05 rms here) that dominates the cross-seed forgetA CI and makes seeds look
# noisier than the fuse actually is. We therefore read out deterministically (T=0) with a
# fresh fixed-seed rng each call, so forgetA reflects the WEIGHTS, not the settling noise.
_EVAL_SEED = 0xE7A1


def _err_on(net, patterns, cfg, rng):
    """Noise-free (T=0) measurement readout of mean reconstruction error over `patterns`.

    `rng` is accepted for call-site compatibility but intentionally NOT used to drive the
    settle: a fresh deterministic eval rng is allocated per call so the measurement is a pure
    function of the learned weights (cuts measurement variance ~4-5x; see README Stage-3)."""
    erng = SeededRNG(_EVAL_SEED)
    tot = 0.0
    for p in patterns:
        for _ in range(cfg.n_settle):
            net.settle_step(erng, T=0.0, clamp_bottom=p)
        net.compute_errors(); tot += float(np.sum(net.eps[0]**2))
    return tot/len(patterns)


def run_continual(use_fuse, seed=0, dim=10, per_task=6, passes=100):
    cfg = _make_cfg(seed, dim)
    rng_proto = np.random.default_rng(seed+5)
    A, B, C = _prototypes(rng_proto, 3, per_task, dim)
    net = PCAreas(cfg); nm = Neuromodulator(cfg); rng = SeededRNG(seed)
    elig = [Eligibility((cfg.dims[l+1],), cfg) for l in range(net.L-1)]
    fuse = [MetaplasticFuse(net.W[l].shape, cfg) for l in range(net.L-1)] if use_fuse else None

    def train(patterns):
        for _ in range(passes):
            for p in patterns:
                for _ in range(cfg.n_settle):
                    net.settle_step(rng, T=cfg.T_floor, clamp_bottom=p)
                    # presynaptic eligibility tracks the latent WHILE it settles to this
                    # pattern, so the four-factor outer product is pattern-specific.
                    for l in range(net.L-1):
                        elig[l].step(a_pre=net.x[l+1])
                net.compute_errors()
                M = nm.update(reward=1.0)
                for l in range(net.L-1):
                    # The fuse REUSES the same Pi, eps, eligibility already computed for
                    # inference — NO Fisher pass, NO task-boundary signal, NO stored anchors.
                    theta = fuse[l].update(net.Pi[l], net.eps[l], elig[l].value) if use_fuse else np.ones_like(net.W[l])
                    net.W[l] += weight_update(M=M, theta=theta, Pi_post=net.Pi[l], eps_post=net.eps[l],
                                              elig=elig[l].value, eta=cfg.eta_w/cfg.tau_w)
                    net.B[l] += (1.0/cfg.tau_b)*feedback_update(net.B[l], a_up=net.x[l+1], eps=net.eps[l], cfg=cfg)
                    net.Pi[l] = precision_update(net.Pi[l], eps_sq=net.eps[l]**2, cfg=cfg)

    train(A); errA_afterA = _err_on(net, A, cfg, rng)
    errC_beforeC = _err_on(net, C, cfg, rng)
    train(B); train(C)
    errA_afterC = _err_on(net, A, cfg, rng); errC_afterC = _err_on(net, C, cfg, rng)
    cbar = float(np.mean([f.c.mean() for f in fuse])) if use_fuse else 0.0
    return {"errA_afterA": errA_afterA, "errA_afterC": errA_afterC,
            "forgetA": errA_afterC - errA_afterA,
            "errC_beforeC": errC_beforeC, "errC_afterC": errC_afterC, "cbar": cbar}
