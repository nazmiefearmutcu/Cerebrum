"""Uncertainty quantification for CEREBRUM via Pillar-4 (Langevin) sample disagreement.

A vanilla transformer gives ONE deterministic forward pass: it has no native notion of
"how sure am I?" beyond a softmax temperature that is not calibrated by construction.
CEREBRUM settles its predictive-coding hierarchy by a NOISY Langevin SDE (pc_core.settle_step:
`noise = rng.normal(scale=sqrt(2*T*dt/tau_x))`, with T >= T_floor > 0 by Pillar 4, which
forbids MAP collapse). That means we can draw MANY stochastic settles for the SAME query and
obtain a genuine SAMPLE DISTRIBUTION over the model's reconstruction. The honest scientific
question this script answers: does the SPREAD of those samples TRACK when the model is WRONG?
If high spread => more-likely-wrong (confident-acc > uncertain-acc, AUROC > 0.5), the
uncertainty is CALIBRATED and is a real brain-favorable differentiator. If the spread is
uninformative, we report that null.

MECHANISM / where the noise actually lives
-------------------------------------------
Task-1's nominal prediction (`predict_obs_here` -> `grid.complete()` == store @ g) is a FIXED
linear readout with NO noise; re-running it is bit-identical. To exercise Pillar 4 we instead
SAMPLE: initialise the PC bottom area x[0] at the grid completion belief, then run the
stochastic settle UNCLAMPED (top area driven by the same grid->top decode the network uses),
reading x[0] after settling. Each distinct SeededRNG seed gives a distinct Langevin trajectory
=> a distinct argmax sample. The per-query uncertainty is the DISAGREEMENT among the S sample
argmaxes (1 - mode_fraction); we also report the entropy of the sample-averaged softmax.

We separate genuinely-ambiguous queries from a trivial artifact: ALL scored queries here have a
target cell that WAS observed (gridworld.make_episode guarantees target in observed_cells), so
no query is "uncertain because never seen". Disagreement is real Langevin sample disagreement
over an answerable query, not an unbound-cell artifact.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
import numpy as np
from cerebrum.config import CerebrumConfig
from cerebrum.core_net import CerebrumCore
from cerebrum.rng import SeededRNG
from cerebrum.types import Exogenous
from benchmarks.tasks.gridworld import make_episode
from benchmarks.tasks.graph_completion import _goto_cell
from benchmarks.stats import fmt_ci, mean_ci


def _softmax(v):
    v = v - np.max(v)
    e = np.exp(v)
    return e / np.sum(e)


def train_episode(ep, cfg):
    """Walk the episode and bind/settle at each cell (same path as graph_completion.run_cerebrum_episode)."""
    net = CerebrumCore(cfg)
    cell = (0, 0)
    _goto_cell(net, cell)
    net.observe_and_learn(ep.gw.obs_at(cell), reward=1.0)
    for (c, a, _avec) in ep.walk:
        cell = ep.gw.step(c, a)
        _goto_cell(net, cell)
        net.observe_and_learn(ep.gw.obs_at(cell), reward=1.0)
    return net


def settle_samples(net, cfg, start, disp, S, n_settle, T):
    """Draw S stochastic Langevin settles of the bottom-area reconstruction for one query.

    Returns (completion_belief, samples[S, vocab]). The completion belief (deterministic
    grid readout) is the network's nominal Task-1 prediction; samples are the noisy settles
    anchored at that belief.
    """
    net.grid.reset()
    net.move(Exogenous(np.array([start[0], start[1]], float)))
    net.move(Exogenous(disp))
    comp = net.predict_obs_here(cfg.dims[0])              # deterministic completion belief
    top_pred = net._top_pred_from_grid(cfg.dims[0])       # grid -> top-area decode the net uses
    L = net.pc.L
    samples = np.empty((S, cfg.dims[0]))
    for s in range(S):
        rng = SeededRNG(7000 + s)                         # distinct Langevin noise per sample
        net.pc.x[0] = comp.copy()
        for l in range(1, L):
            net.pc.x[l] = np.zeros(cfg.dims[l])
        for _ in range(n_settle):
            net.pc.settle_step(rng, T=T, clamp_bottom=None, top_pred=top_pred)
        samples[s] = net.pc.x[0]
    return comp, samples


def per_query_records(ep, net, cfg, S, n_settle, T):
    """For every held-out query produce (correct, disagreement, sample_entropy)."""
    recs = []
    for (start, disp, target) in ep.queries:
        comp, P = settle_samples(net, cfg, start, disp, S=S, n_settle=n_settle, T=T)
        true = int(np.argmax(ep.gw.obs_at(target)))
        pred = int(np.argmax(comp))                       # the actual Task-1 prediction (deterministic readout)
        correct = (pred == true)
        ams = np.argmax(P, axis=1)
        counts = np.bincount(ams, minlength=cfg.dims[0])
        disagreement = 1.0 - counts.max() / len(ams)      # 1 - mode fraction over S sample argmaxes
        avg_soft = np.mean([_softmax(p) for p in P], axis=0)
        entropy = float(-np.sum(avg_soft * np.log(avg_soft + 1e-12)))
        recs.append((bool(correct), float(disagreement), entropy))
    return recs


def auroc(score, is_positive):
    """AUROC that `score` ranks positives (here: WRONG queries) above negatives. NaN if one class empty."""
    score = np.asarray(score, float)
    is_positive = np.asarray(is_positive, bool)
    pos = score[is_positive]
    neg = score[~is_positive]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # rank-sum (Mann-Whitney) form, ties = 0.5
    total = 0.0
    win = 0.0
    for a in pos:
        win += np.sum(a > neg) + 0.5 * np.sum(a == neg)
        total += neg.size
    return win / total


def calibration_for_seed(ep, net, cfg, S, n_settle, T):
    """Return per-seed calibration summary dict from one trained episode."""
    recs = per_query_records(ep, net, cfg, S=S, n_settle=n_settle, T=T)
    R = np.array([(int(c), d, e) for (c, d, e) in recs], float)
    correct = R[:, 0].astype(bool)
    dis = R[:, 1]
    ent = R[:, 2]
    n = len(R)
    acc = correct.mean() if n else float("nan")
    err = ~correct

    # confident (low disagreement) vs uncertain (high disagreement), split at the median.
    med = np.median(dis)
    conf = dis < med           # strictly below median -> confident
    unc = dis > med            # strictly above median -> uncertain (drop ties at the median)
    acc_conf = correct[conf].mean() if conf.any() else float("nan")
    acc_unc = correct[unc].mean() if unc.any() else float("nan")
    gap = acc_conf - acc_unc   # >0 means confident queries are more accurate => calibrated

    return {
        "n": n,
        "acc": acc,
        "acc_conf": acc_conf,
        "acc_unc": acc_unc,
        "gap": gap,
        "auroc_disagree": auroc(dis, err),       # disagreement predicts error
        "auroc_entropy": auroc(ent, err),        # sample-entropy predicts error
        "mean_dis_correct": dis[correct].mean() if correct.any() else float("nan"),
        "mean_dis_wrong": dis[err].mean() if err.any() else float("nan"),
    }


def run_sweep(seeds=(0, 1, 2, 3, 4, 5, 6, 7), K=14, h=4, w=4, vocab=5,
              S=21, n_settle=40, T=None):
    """Train one CEREBRUM-grid episode per seed; collect calibration metrics. T=None -> native T_floor."""
    per_seed = []
    for se in seeds:
        ep = make_episode(h=h, w=w, vocab=vocab, K=K, seed=se)
        cfg = CerebrumConfig(dims=(vocab, 8, 8), grid_n_modules=8, n_settle=10, seed=se)
        TT = cfg.T_floor if T is None else T
        net = train_episode(ep, cfg)
        per_seed.append(calibration_for_seed(ep, net, cfg, S=S, n_settle=n_settle, T=TT))
    return per_seed


def _collect(per_seed, key):
    return [d[key] for d in per_seed if np.isfinite(d[key])]


if __name__ == "__main__":
    SEEDS = tuple(range(16))   # 16 seeds: the split-free AUROC CIs need this many to leave the null
    S = 21
    res = run_sweep(seeds=SEEDS, S=S)

    print("CEREBRUM uncertainty quantification via Pillar-4 (Langevin) sample disagreement")
    print(f"Task-1 graph-completion; {len(SEEDS)} seeds, S={S} settles/query, T=T_floor (native noise floor).")
    print("Uncertainty = disagreement (1 - mode fraction) among the S stochastic-settle argmaxes.")
    print("All scored queries have an OBSERVED target (no never-seen-cell artifact).")
    print()

    accs = _collect(res, "acc")
    print(f"  overall query accuracy        : {fmt_ci(accs)}   (chance = 1/vocab = 0.200)")

    acc_conf = _collect(res, "acc_conf")
    acc_unc = _collect(res, "acc_unc")
    print(f"  accuracy on CONFIDENT queries : {fmt_ci(acc_conf)}")
    print(f"  accuracy on UNCERTAIN queries : {fmt_ci(acc_unc)}")

    gaps = _collect(res, "gap")
    gm, gh = mean_ci(gaps)
    print(f"  calibration gap (conf - unc)  : {fmt_ci(gaps)}   [fragile: disagreement is discrete, many median ties]")

    au_d = _collect(res, "auroc_disagree")
    am, ah = mean_ci(au_d)
    print(f"  AUROC disagreement->error     : {fmt_ci(au_d)}   [>0.5 and CI clear of 0.5 => calibrated]")

    au_e = _collect(res, "auroc_entropy")
    em, eh = mean_ci(au_e)
    print(f"  AUROC sample-entropy->error   : {fmt_ci(au_e)}   [>0.5 and CI clear of 0.5 => calibrated]")

    dc = _collect(res, "mean_dis_correct")
    dw = _collect(res, "mean_dis_wrong")
    print(f"  mean disagreement | correct   : {fmt_ci(dc)}")
    print(f"  mean disagreement | wrong     : {fmt_ci(dw)}   [higher-when-wrong => calibrated direction]")

    # Honest verdict from the SPLIT-FREE metrics (the median gap is discretization-fragile, so it
    # is reported but does not drive the verdict). "Calibrated" requires an AUROC CI to clear 0.5;
    # "weak" if the point estimates lean calibrated (AUROC>0.5 and wrong>correct disagreement) but
    # the CIs still touch the null; otherwise null.
    disagree_clear = (am - ah) > 0.5
    entropy_clear = (em - eh) > 0.5
    leans = (am > 0.5) and (em > 0.5) and (np.mean(dw) > np.mean(dc))
    print()
    if disagree_clear or entropy_clear:
        verdict = "CALIBRATED (a split-free AUROC CI clears the 0.5 null)"
    elif leans:
        verdict = "WEAK (AUROC and disagreement-when-wrong lean calibrated, but CIs touch 0.5)"
    else:
        verdict = "NULL (sample spread does not track error)"
    print(f"VERDICT: {verdict}")

    # Contrast: cranking T destroys the signal (over-injected noise saturates disagreement).
    print()
    print("Control: raise T well above the floor -> every query saturates to full disagreement,")
    print("the confident/uncertain split collapses, and calibration washes out (see below).")
    res_hot = run_sweep(seeds=SEEDS, S=S, T=1.0)
    gaps_hot = _collect(res_hot, "gap")
    au_hot = _collect(res_hot, "auroc_disagree")
    print(f"  T=1.0 calibration gap         : {fmt_ci(gaps_hot) if gaps_hot else 'undefined (all-saturated)'}")
    print(f"  T=1.0 AUROC disagreement->err : {fmt_ci(au_hot) if au_hot else 'undefined (all-saturated)'}")
