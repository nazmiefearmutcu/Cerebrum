import numpy as np
from cerebrum.config import CerebrumConfig
from cerebrum.workspace_net import CerebrumWorkspaceNet
from cerebrum.plasticity import weight_update, precision_update, feedback_update

class SoftWorkspace:
    """ABLATION ONLY — the FORBIDDEN soft write W_j = sum_m P(win_j=m)·read(m). This is a gated
    linear recurrent mixer (linear-attention / Mamba class): the slot becomes a content-conditioned
    superposition of EVERY module's read, broadcast back as a blended top-down prediction. It exists
    here, in benchmarks/baselines, SOLELY as the labeled ablation that proves the strict one-hot write
    in cerebrum/ is load-bearing. It MUST NEVER appear in cerebrum/."""
    def __init__(self, k_slots, content_dim):
        self.k = k_slots; self.dim = content_dim
        self.slots = np.zeros((k_slots, content_dim)); self.last_part = 0.0
    def write_soft(self, P, reads):
        for j in range(self.k):
            self.slots[j] = P[:, j] @ reads                 # SOFT aggregation (the banned move)
        self.last_part = float(np.mean(np.sum(P > 0.05, axis=0)))   # avg # modules contributing per slot
    def broadcast(self):
        return self.slots.sum(axis=0)


def _soft_step(net, cfg, obs, reward):
    """One CerebrumWorkspaceNet.step but with the FORBIDDEN soft write instead of the one-hot write.
    Settling / bidding / selection / module+gate learning are IDENTICAL to net.step and the gate still
    samples a one-hot z; the ONLY changed variable is the WRITE: workspace.write_soft(P) (soft superposition)
    instead of workspace.write(z) (strict one-hot). This isolates write-discreteness as the sole factor."""
    n_modules = net.M_
    bcast = net.workspace.broadcast()
    top = bcast[:net.content_dim] if bcast.size >= net.content_dim else np.zeros(net.content_dim)
    T = net.nm.temperature(0.0)
    errsq = np.zeros(n_modules); reads = np.zeros((n_modules, net.content_dim))
    for mi, mod in enumerate(net.modules):
        for _ in range(cfg.n_settle):
            mod.settle_step(net.rng, T=T, clamp_bottom=obs[mi], top_pred=top)
        mod.compute_errors(top_pred=top)
        errsq[mi] = sum(float(np.sum(e**2)) for e in mod.eps); reads[mi] = mod.x[-1].copy()
    pi = np.array([float(np.mean(m.Pi[-1])) for m in net.modules])
    bids = net.gate.bid(errsq, pi)
    T_gate = cfg.gate_temp if cfg.gate_temp > 0.0 else net.nm.t_gate(max(reward, 1e-3))
    net.gate.select(bids, net.rng, T_gate=T_gate)             # gate still samples one-hot z
    net.workspace.write_soft(net.gate._P, reads)            # but the WRITE uses soft P (the banned move)
    M = net.nm.update(reward)
    for mi, mod in enumerate(net.modules):
        for l in range(mod.L - 1):
            net.elig[mi][l].step(a_pre=mod.x[l + 1])
            mod.W[l] += weight_update(M=M, theta=np.ones_like(mod.W[l]), Pi_post=mod.Pi[l],
                                      eps_post=mod.eps[l], elig=net.elig[mi][l].value,
                                      eta=cfg.eta_w / cfg.tau_w)
            mod.B[l] += (1.0 / cfg.tau_b) * feedback_update(mod.B[l], a_up=mod.x[l + 1], eps=mod.eps[l], cfg=cfg)
            mod.Pi[l] = precision_update(mod.Pi[l], eps_sq=mod.eps[l]**2, cfg=cfg)
    net.gate.learn(M=M); net.gate.homeostasis(M=M)


def run_binding_soft(n_modules=4, k_slots=1, trials=400, seed=0,
                     explore_reward=2.0, reward_scale=5.0, gate_temp=0.1, lam_g=0.05):
    """Soft-mixer ablation of run_binding. Mirrors run_binding EXACTLY (same obs, same reward-scaling /
    gate-temperature regime) and changes ONLY the write rule (one-hot z -> soft superposition over P).

    Routing accuracy is measured the SAME way as the one-hot task: did the write put a CLEAN
    representation of the target into the slot? Formally, routing is correct iff the target owns at
    least `purity` of the slot's write mass. For the one-hot workspace the winner OWNS 100% of the slot,
    so purity is always 1.0 on a routed trial -> this metric is IDENTICAL to run_binding's argmax metric
    (verified: hard argmax_acc == hard purity_acc). For the soft workspace the slot is a superposition
    `sum_m P_m read_m`; whenever the write spreads across >1 module (the gated-SSM behaviour) the target
    fails to reach `purity` and routing degrades. mean_slot_participation reports the avg #modules
    contributing per slot (>1 == continuous mixing, not one-hot). No query-key term anywhere; the bid
    stays scalar own-error."""
    rng = np.random.default_rng(seed)
    # SAME gate knobs as run_binding (lam_g, gate_temp) so the ONLY changed variable is the write rule.
    cfg = CerebrumConfig(dims=(n_modules, n_modules), n_settle=6, seed=seed, lam_g=lam_g, gate_temp=gate_temp)
    net = CerebrumWorkspaceNet(n_modules, k_slots, slice_dim=n_modules, cfg=cfg)
    net.workspace = SoftWorkspace(k_slots, net.content_dim)      # swap in the soft (banned) workspace
    correct = 0; parts = []
    for t in range(trials):
        target = int(rng.integers(0, n_modules))
        obs = [np.zeros(n_modules) for _ in range(n_modules)]
        for m in range(n_modules):
            obs[m][rng.integers(0, n_modules)] = 1.0
        obs[target][:] = 0.0; obs[target][target] = 2.0
        # exploration pass (soft write), judged after seeing the gate's preference
        _soft_step(net, cfg, obs, reward=explore_reward)
        winner = int(np.argmax(net.gate._P[:, 0]))
        reward = reward_scale if winner == target else 0.0
        # learning / measurement pass (soft write)
        _soft_step(net, cfg, obs, reward=reward)
        write_mass = net.gate._P[:, 0]                          # the SOFT write coefficients into the slot
        purity = float(write_mass[target] / (write_mass.sum() + 1e-9))
        routed = purity >= 0.9                                  # target must own a CLEAN (>=90%) slot
        if routed:
            correct += 1
        parts.append(net.workspace.last_part)
    return {"routing_acc": correct / trials,
            "mean_slot_participation": float(np.mean(parts))}
