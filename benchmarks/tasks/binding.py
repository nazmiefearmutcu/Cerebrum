import numpy as np
from grail.config import GRAILConfig
from grail.network2 import GRAILWorkspaceNet

def run_binding(n_modules=4, k_slots=1, trials=400, seed=0,
                explore_reward=2.0, reward_scale=5.0, gate_temp=0.1, lam_g=0.05):
    # Worker-note lever: the scalar own-error bid is already strongly discriminative (the salient
    # target settles to higher ||eps||^2 -> higher bid). The ONLY thing flattening selection is a
    # high T_gate. We LOWER T_gate via reward scaling (t_gate = 1/(|M|+eps)): a non-trivial
    # explore_reward shrinks the exploration-pass temperature so the informative bid drives selection,
    # and a scaled learning reward sharpens the Go weights. No query-key term, bid stays scalar own-error.
    rng = np.random.default_rng(seed)
    # lam_g (Go/NoGo weight decay) prevents spurious preference drift on the random target; gate_temp
    # lowers the selection temperature so the informative scalar bid dominates (still a stochastic
    # one-hot sample). Together these recover most of the near-perfect bid-salience routing the
    # high-temperature drifting gate was throwing away — without any query-key term.
    cfg = GRAILConfig(dims=(n_modules, n_modules), n_settle=6, seed=seed, lam_g=lam_g, gate_temp=gate_temp)
    net = GRAILWorkspaceNet(n_modules, k_slots, slice_dim=n_modules, cfg=cfg)
    wins_per_module = np.zeros(n_modules); correct = 0
    for t in range(trials):
        target = int(rng.integers(0, n_modules))
        obs = [np.zeros(n_modules) for _ in range(n_modules)]
        for m in range(n_modules):
            obs[m][rng.integers(0, n_modules)] = 1.0           # each module sees an object
        obs[target][:] = 0.0; obs[target][target] = 2.0        # target module carries the salient (rewarded) object
        z, _ = net.step(obs, reward=explore_reward)            # exploration pass (low T_gate); reward judged AFTER
        winner = int(np.argmax(z[:, 0]))
        reward = reward_scale if winner == target else 0.0
        # second, learning pass with the actual (scaled) reward signal
        z, _ = net.step(obs, reward=reward)
        winner = int(np.argmax(z[:, 0])); wins_per_module[winner] += 1
        if winner == target: correct += 1
    p = wins_per_module/wins_per_module.sum()
    ent = float(-np.sum(p[p>0]*np.log(p[p>0])))
    return {"routing_acc": correct/trials, "win_entropy": ent, "wins": wins_per_module.tolist()}
