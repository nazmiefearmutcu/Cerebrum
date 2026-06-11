import numpy as np
import torch
from cerebrum.pc_core import PCAreas
from cerebrum.plasticity import Eligibility, weight_update, precision_update, feedback_update
from cerebrum.neuromod import Neuromodulator
from cerebrum.rng import SeededRNG
from benchmarks.tasks.continual import _prototypes, _err_on, _make_cfg

def run_continual_er(seed=0, dim=10, per_task=6, passes=100, buffer_size=50):
    """Experience Replay (ER) baseline. Interleaves previously learned tasks' patterns
    from a replay buffer into current training passes."""
    cfg = _make_cfg(seed, dim)
    A, B, C = _prototypes(np.random.default_rng(seed+5), 3, per_task, dim)
    net = PCAreas(cfg); nm = Neuromodulator(cfg); rng = SeededRNG(seed)
    elig = [Eligibility((cfg.dims[l+1],), cfg) for l in range(net.L-1)]
    
    buffer = []

    def train(patterns, do_replay=True):
        for _ in range(passes):
            for p in patterns:
                # 1. Settle and learn on current pattern
                for _ in range(cfg.n_settle):
                    net.settle_step(rng, T=cfg.T_floor, clamp_bottom=p)
                    for l in range(net.L-1):
                        elig[l].step(a_pre=net.x[l+1])
                net.compute_errors(); M = nm.update(reward=1.0)
                for l in range(net.L-1):
                    dW = weight_update(M=M, theta=torch.ones_like(net.W[l]), Pi_post=net.Pi[l],
                                       eps_post=net.eps[l], elig=elig[l].value, eta=cfg.eta_w/cfg.tau_w)
                    net.W[l] += dW
                    net.B[l] += (1.0/cfg.tau_b)*feedback_update(net.B[l], a_up=net.x[l+1], eps=net.eps[l], cfg=cfg)
                    net.Pi[l] = precision_update(net.Pi[l], eps_sq=net.eps[l]**2, cfg=cfg)
                
                # 2. Interleave replay step if buffer is not empty
                if do_replay and len(buffer) > 0:
                    idx = int(rng._rng.integers(0, len(buffer)))
                    p_rep = buffer[idx]
                    for _ in range(cfg.n_settle):
                        net.settle_step(rng, T=cfg.T_floor, clamp_bottom=p_rep)
                        for l in range(net.L-1):
                            elig[l].step(a_pre=net.x[l+1])
                    net.compute_errors(); M = nm.update(reward=1.0)
                    for l in range(net.L-1):
                        dW = weight_update(M=M, theta=torch.ones_like(net.W[l]), Pi_post=net.Pi[l],
                                           eps_post=net.eps[l], elig=elig[l].value, eta=cfg.eta_w/cfg.tau_w)
                        net.W[l] += dW
                        net.B[l] += (1.0/cfg.tau_b)*feedback_update(net.B[l], a_up=net.x[l+1], eps=net.eps[l], cfg=cfg)
                        net.Pi[l] = precision_update(net.Pi[l], eps_sq=net.eps[l]**2, cfg=cfg)

    # Train Task A without replay (buffer is empty anyway)
    train(A, do_replay=False)
    errA_afterA = _err_on(net, A, cfg, rng); errC_beforeC = _err_on(net, C, cfg, rng)
    
    # Store Task A patterns
    for p in A:
        if len(buffer) < buffer_size:
            buffer.append(p)
            
    # Train Task B with replay, then store Task B patterns
    train(B, do_replay=True)
    for p in B:
        if len(buffer) < buffer_size:
            buffer.append(p)
            
    # Train Task C with replay
    train(C, do_replay=True)
    errA_afterC = _err_on(net, A, cfg, rng); errC_afterC = _err_on(net, C, cfg, rng)
    
    return {"errA_afterA": errA_afterA, "errA_afterC": errA_afterC, "forgetA": errA_afterC-errA_afterA,
            "errC_beforeC": errC_beforeC, "errC_afterC": errC_afterC, "used_replay": True}


def run_continual_der(seed=0, dim=10, per_task=6, passes=100, buffer_size=50, alpha=0.5):
    """Dark Experience Replay (DER++) baseline. Replays patterns and penalizes deviation
    from stored latent predictions (knowledge distillation)."""
    cfg = _make_cfg(seed, dim)
    A, B, C = _prototypes(np.random.default_rng(seed+5), 3, per_task, dim)
    net = PCAreas(cfg); nm = Neuromodulator(cfg); rng = SeededRNG(seed)
    elig = [Eligibility((cfg.dims[l+1],), cfg) for l in range(net.L-1)]
    
    # Replay buffer stores tuple: (pattern, latent_prediction)
    buffer = []

    def train(patterns, do_replay=True):
        for _ in range(passes):
            for p in patterns:
                # 1. Settle and learn on current pattern
                for _ in range(cfg.n_settle):
                    net.settle_step(rng, T=cfg.T_floor, clamp_bottom=p)
                    for l in range(net.L-1):
                        elig[l].step(a_pre=net.x[l+1])
                net.compute_errors(); M = nm.update(reward=1.0)
                for l in range(net.L-1):
                    dW = weight_update(M=M, theta=torch.ones_like(net.W[l]), Pi_post=net.Pi[l],
                                       eps_post=net.eps[l], elig=elig[l].value, eta=cfg.eta_w/cfg.tau_w)
                    net.W[l] += dW
                    net.B[l] += (1.0/cfg.tau_b)*feedback_update(net.B[l], a_up=net.x[l+1], eps=net.eps[l], cfg=cfg)
                    net.Pi[l] = precision_update(net.Pi[l], eps_sq=net.eps[l]**2, cfg=cfg)
                
                # 2. Interleave DER++ replay step if buffer is not empty
                if do_replay and len(buffer) > 0:
                    idx = int(rng._rng.integers(0, len(buffer)))
                    p_rep, z_rep = buffer[idx]
                    
                    # Settle on the replayed pattern
                    for _ in range(cfg.n_settle):
                        net.settle_step(rng, T=cfg.T_floor, clamp_bottom=p_rep)
                        for l in range(net.L-1):
                            elig[l].step(a_pre=net.x[l+1])
                    
                    net.compute_errors()
                    
                    # Apply DER++ penalty: add mismatch between current top latent and stored top latent
                    # to top error area.
                    with torch.no_grad():
                        mismatch = net.x[-1] - z_rep
                        net.eps[-1] += alpha * mismatch
                    
                    M = nm.update(reward=1.0)
                    for l in range(net.L-1):
                        dW = weight_update(M=M, theta=torch.ones_like(net.W[l]), Pi_post=net.Pi[l],
                                           eps_post=net.eps[l], elig=elig[l].value, eta=cfg.eta_w/cfg.tau_w)
                        net.W[l] += dW
                        net.B[l] += (1.0/cfg.tau_b)*feedback_update(net.B[l], a_up=net.x[l+1], eps=net.eps[l], cfg=cfg)
                        net.Pi[l] = precision_update(net.Pi[l], eps_sq=net.eps[l]**2, cfg=cfg)

    # Helper function to get top prediction for patterns to store in buffer
    def get_latents(patterns):
        latents = []
        for p in patterns:
            for _ in range(cfg.n_settle):
                net.settle_step(rng, T=0.0, clamp_bottom=p)
            latents.append(net.x[-1].clone())
        return latents

    # Train Task A without replay
    train(A, do_replay=False)
    errA_afterA = _err_on(net, A, cfg, rng); errC_beforeC = _err_on(net, C, cfg, rng)
    
    # Store Task A patterns + latents
    latents_A = get_latents(A)
    for p, z in zip(A, latents_A):
        if len(buffer) < buffer_size:
            buffer.append((p, z))
            
    # Train Task B with replay
    train(B, do_replay=True)
    latents_B = get_latents(B)
    for p, z in zip(B, latents_B):
        if len(buffer) < buffer_size:
            buffer.append((p, z))
            
    # Train Task C with replay
    train(C, do_replay=True)
    errA_afterC = _err_on(net, A, cfg, rng); errC_afterC = _err_on(net, C, cfg, rng)
    
    return {"errA_afterA": errA_afterA, "errA_afterC": errA_afterC, "forgetA": errA_afterC-errA_afterA,
            "errC_beforeC": errC_beforeC, "errC_afterC": errC_afterC, "used_der": True}
