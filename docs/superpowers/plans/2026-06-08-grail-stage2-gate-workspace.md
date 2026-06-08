# GRAIL Stage 2 Implementation Plan — Gate + Workspace + Broadcast (the emergent mixer)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes.

**Goal:** Build the stochastic basal-ganglia gate, the k≪n workspace, and the thalamo-cortical broadcast loop so that information routing between cortical modules EMERGES from `bid → one-hot write → broadcast → reshape-next-bid` — with NO attention matrix, NO delta rule, NO state-space operator — and prove the strict one-hot write is load-bearing (relaxing it to soft weights collapses to a gated-SSM).

**Architecture:** Builds on Stage 0+1 (`grail/pc_core.py`, `grail/plasticity.py`, `grail/neuromod.py`, `grail/invariants.py`, `grail/counters.py`). Multiple `PCAreas` modules each settle on their own input slice; each emits a SCALAR own-error bid; a striatal Go/NoGo gate picks a stochastic strict-one-hot winner per workspace slot; winners write their content; the workspace broadcasts back as a top-down prediction (efference copy) that re-enters each module's settling. Gate weights learn by a LOCAL three-factor rule gated by the scalar neuromodulator M (no backprop through the discrete decision). Pure NumPy, no autograd.

**Tech Stack:** Python 3, numpy, pytest. No torch/jax/sklearn in `grail/`.

**Spec:** `docs/superpowers/specs/2026-06-08-grail-cortical-workspace-design.md` §3② (gate), §10 (soft-mixing ablation), §12 (invariants).

**Bans (unchanged, enforced):** the bid `b_m` is a SCALAR own-error salience — NEVER a cross-module query-key dot product. Selection is a strict one-hot SAMPLE — NEVER argmax, NEVER soft-weighted aggregation (`W_j ← Σ_m P·read(m)` is FORBIDDEN, BAN-1). Gate learning uses only the scalar M + local eligibility — never a global error vector (BAN-2). No autograd.

---

## File Structure (additions)

```
grail/
  gate.py        # BasalGangliaGate: scalar bids, striatal Go/NoGo, stochastic one-hot select, local 3-factor learn, dead-expert homeostasis
  workspace.py   # Workspace: k slots, strict one-hot write, broadcast (efference copy)
  network2.py    # GRAILWorkspaceNet: M modules + gate + workspace + broadcast loop
tests/
  test_gate.py  test_workspace.py  test_network2.py  test_stage2_smoke.py
benchmarks/
  tasks/binding.py            # multi-module selective-routing task + metric
  baselines/soft_mixer.py     # soft-weight relaxation = gated-SSM ablation
  run_stage2.py               # routing emergence + load balance + one-hot-vs-soft demonstration
```

---

## Task 1: Workspace — strict one-hot write + broadcast

**Files:** Create `grail/workspace.py`, `tests/test_workspace.py`

- [ ] **Step 1: Failing test** `tests/test_workspace.py`:
```python
import numpy as np, pytest
from grail.workspace import Workspace

def test_one_hot_write_takes_winner_content():
    ws = Workspace(k_slots=2, content_dim=3)
    z = np.array([[1.0,0.0],[0.0,1.0],[0.0,0.0]])      # module0 -> slot0, module1 -> slot1
    reads = np.array([[1.,1.,1.],[2.,2.,2.],[9.,9.,9.]])
    ws.write(z, reads)
    assert np.allclose(ws.slots[0], [1,1,1]) and np.allclose(ws.slots[1], [2,2,2])

def test_soft_weights_are_rejected():
    ws = Workspace(k_slots=1, content_dim=2)
    zsoft = np.array([[0.6],[0.4]])                    # soft mixing weights = BAN-1
    with pytest.raises(AssertionError):
        ws.write(zsoft, np.array([[1.,0.],[0.,1.]]))

def test_broadcast_sums_slot_contents():
    ws = Workspace(k_slots=2, content_dim=2)
    ws.slots[0] = [1.,0.]; ws.slots[1] = [0.,3.]
    assert np.allclose(ws.broadcast(), [1.,3.])
```

- [ ] **Step 2: Run, fail. Step 3: Implement** `grail/workspace.py`:
```python
import numpy as np
from .invariants import assert_one_hot

class Workspace:
    """k<<n latent slots. STRICT one-hot write (slot j content = the single winning module's read);
    soft-weighted aggregation is FORBIDDEN (BAN-1). Broadcast returns slot contents to all modules
    as a top-down prediction (efference copy)."""
    def __init__(self, k_slots, content_dim):
        self.k = k_slots; self.dim = content_dim
        self.slots = np.zeros((k_slots, content_dim))
    def write(self, z, reads):
        assert_one_hot(z, axis=0)                       # raises on soft weights
        for j in range(self.k):
            winners = np.flatnonzero(z[:, j] > 0.5)
            if winners.size:
                self.slots[j] = reads[winners[0]]        # strict one-hot read of the winner
    def broadcast(self):
        return self.slots.sum(axis=0)
```
- [ ] **Step 4: PASS. Step 5: Commit** — `git commit -am "feat(stage2): workspace strict one-hot write + broadcast"`

---

## Task 2: BasalGangliaGate — scalar bids + stochastic one-hot selection

**Files:** Create `grail/gate.py`, `tests/test_gate.py`

Spec §3②. `b_m = π_m·E[‖ε_m‖²] + θ_m` (scalar, own-error only). `u_mj = G_mj b_m − Σ_{m'≠m} N_{m'j} b_{m'}`. `P(win_j=m)=softmax(u_mj/T_gate + Gumbel)`, `z` = one-hot sample (argmax over Gumbel-perturbed logits = exact softmax sample).

- [ ] **Step 1: Failing test** `tests/test_gate.py`:
```python
import numpy as np
from grail.config import GRAILConfig
from grail.gate import BasalGangliaGate
from grail.rng import SeededRNG
from grail.invariants import assert_one_hot

def test_bid_is_scalar_own_error_plus_excitability():
    g = BasalGangliaGate(n_modules=3, k_slots=2, cfg=GRAILConfig(), seed=0)
    bids = g.bid(err_sq=np.array([1.0,4.0,0.0]), pi=np.array([1.0,1.0,1.0]))
    assert bids.shape == (3,)
    assert bids[1] > bids[0] > bids[2]                  # higher own-error -> higher bid (no cross-module term)

def test_selection_is_one_hot_per_slot():
    g = BasalGangliaGate(n_modules=4, k_slots=2, cfg=GRAILConfig(), seed=1)
    z = g.select(bids=np.array([1.,2.,0.5,0.1]), rng=SeededRNG(0), T_gate=0.5)
    assert z.shape == (4,2); assert_one_hot(z, axis=0)

def test_selection_is_stochastic_not_argmax():
    g = BasalGangliaGate(n_modules=3, k_slots=1, cfg=GRAILConfig(), seed=2)
    wins = [int(np.argmax(g.select(np.array([1.0,0.9,0.8]), SeededRNG(s), T_gate=1.0)[:,0])) for s in range(50)]
    assert len(set(wins)) > 1                            # noise -> not always the top bidder (Pillar 4)
```

- [ ] **Step 2: Fail. Step 3: Implement** `grail/gate.py`:
```python
import numpy as np
from .invariants import assert_one_hot, assert_scalar_M

class BasalGangliaGate:
    """Stochastic basal-ganglia gate. Modules bid a SCALAR own-error salience for k workspace slots;
    a striatal Go/NoGo competition selects a strict one-hot winner per slot WITH noise (never argmax,
    never soft). Gate weights learn by a LOCAL three-factor rule gated by the scalar neuromodulator M.
    There is NO query-key / content-similarity term anywhere — the competition can never become attention."""
    def __init__(self, n_modules, k_slots, cfg, seed=0):
        self.M_ = n_modules; self.k = k_slots; self.cfg = cfg
        rng = np.random.default_rng(seed + 31)
        self.G = 0.5 + 0.1*rng.standard_normal((n_modules, k_slots))   # Go weights
        self.N = 0.1*rng.standard_normal((n_modules, k_slots))         # NoGo weights
        self.theta = np.zeros(n_modules)                               # dead-expert excitability
        self._P = None; self._z = None; self._bid = None

    def bid(self, err_sq, pi):
        return pi*np.asarray(err_sq) + self.theta                       # (M,) scalar per module

    def select(self, bids, rng, T_gate):
        bids = np.asarray(bids, float)
        z = np.zeros((self.M_, self.k)); P = np.zeros((self.M_, self.k))
        for j in range(self.k):
            inhib_total = float(np.sum(self.N[:, j]*bids))
            u = self.G[:, j]*bids - (inhib_total - self.N[:, j]*bids)   # u_mj = G b_m - sum_{m'!=m} N b_m'
            logits = u/max(T_gate,1e-6) + rng.gumbel((self.M_,))
            ex = np.exp(logits - logits.max()); P[:, j] = ex/ex.sum()
            z[int(np.argmax(logits)), j] = 1.0                          # Gumbel-argmax = exact softmax SAMPLE
        assert_one_hot(z, axis=0)
        self._P, self._z, self._bid = P, z, bids
        return z

    def learn(self, M, eta=None):
        assert_scalar_M(M)
        eta = self.cfg.eta_w if eta is None else eta
        e = (self._z - self._P) * self._bid[:, None]                    # local 3-factor eligibility
        self.G += eta*M*e
        self.N += -eta*M*e                                              # NoGo opponent (opposite sign)

    def homeostasis(self, gamma_up=0.02, gamma_dn=0.05):
        wins = np.minimum(self._z.sum(axis=1), 1.0)
        self.theta += gamma_up*(1.0 - wins) - gamma_dn*wins             # rises on loss, falls on win
```
- [ ] **Step 4: PASS. Step 5: Commit** — `git commit -am "feat(stage2): basal-ganglia gate (scalar bids, stochastic one-hot select)"`

---

## Task 3: Gate local learning + dead-expert homeostasis (behavioral tests)

**Files:** add tests to `tests/test_gate.py`

- [ ] **Step 1: Failing tests (append):**
```python
def test_local_learning_raises_win_prob_of_rewarded_module():
    g = BasalGangliaGate(n_modules=3, k_slots=1, cfg=GRAILConfig(eta_w=0.5), seed=3)
    rng = SeededRNG(0)
    # repeatedly: module 1 wins and is rewarded (M>0) -> its Go weight should grow
    G1_before = g.G[1,0]
    for _ in range(30):
        g.select(np.array([0.5,1.0,0.5]), rng, T_gate=0.5)
        g.learn(M=1.0)
    assert g.G[1,0] > G1_before                          # rewarded winner's Go weight increases (scalar M)

def test_homeostasis_raises_excitability_of_starved_module():
    g = BasalGangliaGate(n_modules=3, k_slots=1, cfg=GRAILConfig(), seed=4)
    rng = SeededRNG(1)
    for _ in range(40):
        g.select(np.array([5.0, 0.0, 0.0]), rng, T_gate=0.2)  # module 0 hogs the slot
        g.homeostasis()
    assert g.theta[1] > 0 and g.theta[2] > 0              # starved modules' excitability rises (anti-dead-expert)
    assert g.theta[0] < g.theta[1]                        # the hog's excitability is suppressed
```
- [ ] **Step 2: Fail (learn/homeostasis already implemented in Task 2 — these are behavioral checks). If they fail, fix the rule, not the test.** **Step 3:** confirm `learn`/`homeostasis` from Task 2 produce these behaviors; adjust signs/rates if needed.
- [ ] **Step 4: PASS. Step 5: Commit** — `git commit -am "test(stage2): gate local learning + dead-expert homeostasis behavior"`

---

## Task 4: GRAILWorkspaceNet — multi-module + gate + workspace + broadcast loop

**Files:** Create `grail/network2.py`, `tests/test_network2.py`

M cortical modules, each a `PCAreas` on its own input slice; per step: settle each module with the current broadcast as top-down; compute each module's `err_sq = Σ‖ε‖²`; bid; select one-hot winners; write winners' top-area content to slots; broadcast back; learn (module plasticity + gate learning + homeostasis). The broadcast re-entering settling is the seed of emergent routing.

- [ ] **Step 1: Failing test** `tests/test_network2.py`:
```python
import numpy as np
from grail.config import GRAILConfig
from grail.network2 import GRAILWorkspaceNet
from grail.invariants import assert_one_hot

def test_step_runs_routes_and_counts():
    cfg = GRAILConfig(dims=(4,4), n_settle=8, seed=0)
    net = GRAILWorkspaceNet(n_modules=3, k_slots=2, slice_dim=4, cfg=cfg)
    obs = [np.array([1.,0,0,0]), np.array([0,1.,0,0]), np.array([0,0,1.,0])]
    z, M = net.step(obs, reward=1.0)
    assert_one_hot(z, axis=0)                            # routing decision is one-hot
    assert np.ndim(M) == 0                               # scalar neuromodulator
    assert net.counters.global_comm_infer > 0           # broadcast vectors counted at infer time

def test_broadcast_influences_modules():
    cfg = GRAILConfig(dims=(4,4), n_settle=8, seed=1)
    net = GRAILWorkspaceNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg)
    obs = [np.array([1.,0,0,0]), np.array([0,0,0,1.])]
    net.step(obs, reward=1.0)
    assert np.any(net.workspace.slots != 0)              # a winner wrote content that will broadcast next step
```

- [ ] **Step 2: Fail. Step 3: Implement** `grail/network2.py`:
```python
import numpy as np
from .pc_core import PCAreas
from .gate import BasalGangliaGate
from .workspace import Workspace
from .neuromod import Neuromodulator
from .plasticity import Eligibility, weight_update, precision_update, feedback_update
from .counters import Counters
from .rng import SeededRNG
from .invariants import assert_scalar_M

class GRAILWorkspaceNet:
    """Stage-2 cortical workspace network: M modules compete via a stochastic gate for k slots;
    winners' content is broadcast back as top-down prediction. Routing EMERGES from the loop;
    there is no attention/mixer module."""
    def __init__(self, n_modules, k_slots, slice_dim, cfg):
        self.cfg = cfg; self.M_ = n_modules; self.k = k_slots
        # each module is a PCAreas whose bottom area = its input slice
        mdims = (slice_dim,) + tuple(cfg.dims[1:]) if len(cfg.dims) > 1 else (slice_dim, slice_dim)
        from dataclasses import replace
        self.modules = [PCAreas(replace(cfg, dims=mdims, seed=cfg.seed+i)) for i in range(n_modules)]
        self.content_dim = mdims[-1]
        self.gate = BasalGangliaGate(n_modules, k_slots, cfg, seed=cfg.seed)
        self.workspace = Workspace(k_slots, self.content_dim)
        self.nm = Neuromodulator(cfg)
        self.rng = SeededRNG(cfg.seed)
        self.counters = Counters()
        self.elig = [[Eligibility((m.cfg.dims[l+1],), cfg) for l in range(m.L-1)] for m in self.modules]

    def step(self, obs_slices, reward):
        bcast = self.workspace.broadcast()                          # top-down efference copy from last step
        self.counters.record_global_infer_vectors(k=self.k, width=self.content_dim)
        # top-down prediction to each module's top area = a frozen projection of the broadcast
        top_pred = bcast[:self.content_dim] if bcast.size >= self.content_dim else np.zeros(self.content_dim)
        # 1) settle every module with the broadcast as top-down
        T = self.nm.temperature(0.0)
        err_sq = np.zeros(self.M_); reads = np.zeros((self.M_, self.content_dim))
        for m_i, mod in enumerate(self.modules):
            for _ in range(self.cfg.n_settle):
                mod.settle_step(self.rng, T=T, clamp_bottom=obs_slices[m_i],
                                top_pred=top_pred, counters=self.counters)
            mod.compute_errors(top_pred=top_pred)
            err_sq[m_i] = sum(float(np.sum(e**2)) for e in mod.eps)
            reads[m_i] = mod.x[-1].copy()                            # module content = top-area activity
        # 2) gate: bid (scalar own-error) -> stochastic one-hot select -> write -> broadcast
        pi = np.array([float(np.mean(mod.Pi[-1])) for mod in self.modules])
        bids = self.gate.bid(err_sq=err_sq, pi=pi)
        z = self.gate.select(bids, self.rng, T_gate=self.nm.t_gate(max(reward,1e-3)))
        self.workspace.write(z, reads)
        # 3) learn: scalar M gates module plasticity + gate learning + homeostasis
        M = self.nm.update(reward); assert_scalar_M(M); self.counters.record_global_learn(1)
        for m_i, mod in enumerate(self.modules):
            for l in range(mod.L-1):
                self.elig[m_i][l].step(a_pre=mod.x[l+1])
                mod.W[l] += weight_update(M=M, theta=np.ones_like(mod.W[l]), Pi_post=mod.Pi[l],
                                          eps_post=mod.eps[l], elig=self.elig[m_i][l].value,
                                          eta=self.cfg.eta_w/self.cfg.tau_w)
                mod.B[l] += (1.0/self.cfg.tau_b)*feedback_update(mod.B[l], a_up=mod.x[l+1], eps=mod.eps[l], cfg=self.cfg)
                mod.Pi[l] = precision_update(mod.Pi[l], eps_sq=mod.eps[l]**2, cfg=self.cfg)
        self.gate.learn(M=M); self.gate.homeostasis()
        return z, M
```
> **Worker note:** keep `mdims` consistent: a module's bottom area dim = `slice_dim`, higher areas reuse `cfg.dims[1:]`. The top-down `top_pred` must match the module top-area dim (`content_dim`). If a shape mismatch arises, align `top_pred` length to `content_dim` (slice/pad the broadcast) — do NOT route module state into the broadcast in a data-dependent way that would make routing a learned mixing matrix; the broadcast is content written by one-hot winners only.

- [ ] **Step 4: PASS. Step 5: Commit** — `git commit -am "feat(stage2): GRAILWorkspaceNet (modules + gate + workspace + broadcast loop)"`

---

## Task 5: Binding/routing task + emergence metric

**Files:** Create `benchmarks/tasks/binding.py`, `tests/test_stage2_smoke.py`

A selective-routing task: `M` modules each receive a one-hot "object"; one designated TARGET module per trial carries the rewarded object. The gate must learn to route the target module into a slot. Metric: fraction of trials where the target module wins a slot (routing accuracy), and the win-entropy across modules (load balance — should stay > 0, no permanently dead experts).

- [ ] **Step 1: Failing test** `tests/test_stage2_smoke.py`:
```python
import numpy as np
from benchmarks.tasks.binding import run_binding

def test_routing_accuracy_rises_above_chance():
    res = run_binding(n_modules=4, k_slots=1, trials=400, seed=0)
    assert res["routing_acc"] > 1.0/4 + 0.1            # gate learns to route the target above chance
    assert res["win_entropy"] > 0.1                     # load is balanced (no single dead/hog collapse)
```

- [ ] **Step 2: Fail. Step 3: Implement** `benchmarks/tasks/binding.py`:
```python
import numpy as np
from grail.config import GRAILConfig
from grail.network2 import GRAILWorkspaceNet

def run_binding(n_modules=4, k_slots=1, trials=400, seed=0):
    rng = np.random.default_rng(seed)
    cfg = GRAILConfig(dims=(n_modules, n_modules), n_settle=6, seed=seed)
    net = GRAILWorkspaceNet(n_modules, k_slots, slice_dim=n_modules, cfg=cfg)
    wins_per_module = np.zeros(n_modules); correct = 0
    for t in range(trials):
        target = int(rng.integers(0, n_modules))
        obs = [np.zeros(n_modules) for _ in range(n_modules)]
        for m in range(n_modules):
            obs[m][rng.integers(0, n_modules)] = 1.0           # each module sees an object
        obs[target][:] = 0.0; obs[target][target] = 2.0        # target module carries the salient (rewarded) object
        z, _ = net.step(obs, reward=0.0)                       # reward assigned AFTER seeing who won
        winner = int(np.argmax(z[:, 0]))
        reward = 1.0 if winner == target else 0.0
        # second, learning pass with the actual reward signal
        z, _ = net.step(obs, reward=reward)
        winner = int(np.argmax(z[:, 0])); wins_per_module[winner] += 1
        if winner == target: correct += 1
    p = wins_per_module/wins_per_module.sum()
    ent = float(-np.sum(p[p>0]*np.log(p[p>0])))
    return {"routing_acc": correct/trials, "win_entropy": ent, "wins": wins_per_module.tolist()}
```
> **Worker note:** if routing_acc does not rise, raise `cfg.eta_w` (gate learning rate) or lower `T_gate` via reward scaling; the salient target (value 2.0 → higher own-error after settling → higher bid) plus M-gated Go-weight learning should make the target win more often. Do NOT introduce a query-key term to "help" — that is a ban violation. Keep the bid scalar own-error.

- [ ] **Step 4: PASS. Step 5: Commit** — `git commit -am "feat(stage2): selective-routing task + emergence/load-balance metric"`

---

## Task 6: Soft-mixer ablation — prove one-hot is load-bearing (BAN-1 enforcement)

**Files:** Create `benchmarks/baselines/soft_mixer.py`; add test to `tests/test_stage2_smoke.py`

Replace the strict one-hot write with the FORBIDDEN soft aggregation `W_j ← Σ_m P(win_j=m)·read(m)` and show it becomes a content-conditioned write into a bounded recurrent state with a fixed read-out — i.e. a gated linear recurrent mixer (linear-attention/Mamba class). Demonstrate the routing metric degrades toward undifferentiated mixing (entropy → max, routing_acc → chance), proving discreteness is load-bearing, not cosmetic.

- [ ] **Step 1: Failing test (append):**
```python
def test_soft_mixer_collapses_to_undifferentiated_mixing():
    from benchmarks.baselines.soft_mixer import run_binding_soft
    from benchmarks.tasks.binding import run_binding
    hard = run_binding(n_modules=4, k_slots=1, trials=400, seed=0)
    soft = run_binding_soft(n_modules=4, k_slots=1, trials=400, seed=0)
    # soft aggregation mixes all modules' content every step -> cannot route selectively
    assert soft["routing_acc"] < hard["routing_acc"]
    assert soft["mean_slot_participation"] > 1.5        # >1 module contributes per slot (continuous mixing, not one-hot)
```

- [ ] **Step 2: Fail. Step 3: Implement** `benchmarks/baselines/soft_mixer.py`:
```python
import numpy as np
from grail.config import GRAILConfig
from grail.network2 import GRAILWorkspaceNet

class SoftWorkspace:
    """ABLATION ONLY — the FORBIDDEN soft write W_j = sum_m P(win_j=m) read(m). This is a gated
    linear recurrent mixer (linear-attention/Mamba class). Used solely to prove one-hot is load-bearing."""
    def __init__(self, k_slots, content_dim):
        self.k=k_slots; self.dim=content_dim; self.slots=np.zeros((k_slots,content_dim)); self.last_part=0.0
    def write_soft(self, P, reads):
        for j in range(self.k):
            self.slots[j] = P[:, j] @ reads                 # SOFT aggregation (the banned move)
        self.last_part = float(np.mean(np.sum(P > 0.05, axis=0)))   # avg # modules contributing per slot
    def broadcast(self): return self.slots.sum(axis=0)

def run_binding_soft(n_modules=4, k_slots=1, trials=400, seed=0):
    rng = np.random.default_rng(seed)
    cfg = GRAILConfig(dims=(n_modules,n_modules), n_settle=6, seed=seed)
    net = GRAILWorkspaceNet(n_modules, k_slots, slice_dim=n_modules, cfg=cfg)
    net.workspace = SoftWorkspace(k_slots, net.content_dim)      # swap in the soft (banned) workspace
    # monkeypatch the write path: use P (soft) instead of z (one-hot)
    wins=np.zeros(n_modules); correct=0; parts=[]
    for t in range(trials):
        target=int(rng.integers(0,n_modules))
        obs=[np.zeros(n_modules) for _ in range(n_modules)]
        for m in range(n_modules): obs[m][rng.integers(0,n_modules)]=1.0
        obs[target][:]=0.0; obs[target][target]=2.0
        # replicate net.step but with soft write
        bcast=net.workspace.broadcast(); top=bcast[:net.content_dim] if bcast.size>=net.content_dim else np.zeros(net.content_dim)
        errsq=np.zeros(n_modules); reads=np.zeros((n_modules,net.content_dim))
        for mi,mod in enumerate(net.modules):
            for _ in range(cfg.n_settle): mod.settle_step(net.rng,T=0.05,clamp_bottom=obs[mi],top_pred=top)
            mod.compute_errors(top_pred=top); errsq[mi]=sum(float(np.sum(e**2)) for e in mod.eps); reads[mi]=mod.x[-1].copy()
        pi=np.array([float(np.mean(m.Pi[-1])) for m in net.modules])
        bids=net.gate.bid(errsq,pi); z=net.gate.select(bids,net.rng,T_gate=1.0)
        net.workspace.write_soft(net.gate._P, reads)            # SOFT write
        winner=int(np.argmax(net.gate._P[:,0])); reward=1.0 if winner==target else 0.0
        net.gate.learn(M=net.nm.update(reward)); net.gate.homeostasis()
        wins[winner]+=1; parts.append(net.workspace.last_part)
        if winner==target: correct+=1
    return {"routing_acc": correct/trials, "mean_slot_participation": float(np.mean(parts))}
```
- [ ] **Step 4: PASS. Step 5: Commit** — `git commit -am "feat(stage2): soft-mixer ablation proves one-hot write is load-bearing"`

---

## Task 7: run_stage2 demonstration + README update + full suite green

**Files:** Create `benchmarks/run_stage2.py`; update `README.md`; run full suite.

- [ ] **Step 1:** `benchmarks/run_stage2.py` prints: routing accuracy + win-entropy (one-hot) vs routing accuracy + slot-participation (soft), with a sys.path bootstrap like run_task1.py. Demonstrates: (a) emergent routing above chance, (b) load balance, (c) soft collapses to mixing.
```python
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.tasks.binding import run_binding
from benchmarks.baselines.soft_mixer import run_binding_soft

if __name__ == "__main__":
    for nm in (4, 6):
        hard = run_binding(n_modules=nm, k_slots=1, trials=500, seed=0)
        soft = run_binding_soft(n_modules=nm, k_slots=1, trials=500, seed=0)
        print(f"[M={nm}] one-hot: routing_acc={hard['routing_acc']:.3f} entropy={hard['win_entropy']:.3f} "
              f"(chance={1.0/nm:.3f}) | soft: routing_acc={soft['routing_acc']:.3f} "
              f"participation={soft['mean_slot_participation']:.2f}")
```
- [ ] **Step 2:** Run `python3 benchmarks/run_stage2.py`, capture the table.
- [ ] **Step 3:** Update `README.md` with a Stage-2 section: the emergent-mixer claim, the one-hot-vs-soft result, and reiterate the honesty gate (still zero open problems solved; this stage demonstrates emergent routing, NOT scaling parity).
- [ ] **Step 4:** Run full suite `cd /Users/nazmi/grail && python3 -m pytest -q` (all green).
- [ ] **Step 5: Commit** — `git commit -am "docs(stage2): README + emergent routing demonstration; full suite green"`

---

## Self-Review (planner)

**Spec coverage:** §3② bid (scalar own-error, no Q·K) → Task 2; striatal Go/NoGo + stochastic one-hot → Task 2; local 3-factor gate learning (scalar M) → Tasks 2-3; dead-expert homeostasis → Tasks 2-3; workspace strict one-hot write → Task 1; broadcast/efference copy → Tasks 1,4; emergent routing (no attention matrix) → Tasks 4-5; §10 soft-mixing ablation collapses to gated-SSM → Task 6; §12 invariants (one-hot via assert_one_hot, scalar M via assert_scalar_M) → Tasks 1-2,4. ✓

**Placeholder scan:** all code steps have complete code or behavioral-contract tests; two worker notes flag the known tricky spots (module dim consistency; routing learning rate) without leaving logic unspecified. ✓

**Type consistency:** `Workspace(k_slots, content_dim).write(z, reads)/broadcast()`; `BasalGangliaGate(n_modules,k_slots,cfg,seed).bid(err_sq,pi)/select(bids,rng,T_gate)/learn(M,eta)/homeostasis()`; `GRAILWorkspaceNet(n_modules,k_slots,slice_dim,cfg).step(obs_slices,reward)` are consistent across Tasks 1-7. Reuses Stage-1 `PCAreas`, `weight_update`, `precision_update`, `feedback_update`, `Neuromodulator`, `Counters`, `SeededRNG` with their existing signatures. ✓

**Intentional scope gap:** Stage 3 (metaplastic fuse + catastrophic-forgetting Task-2) is the next plan.
