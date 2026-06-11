"""PRINCIPLED FACTORIZATION PROBE — does the LOCAL rule build a compositionally-GENERALIZING
factored latent? (Corrects an earlier overstated negative.)

WHY THIS EXISTS — correcting an overstated null
-----------------------------------------------
The earlier compositional "f1->f2 completion" probe (benchmarks/tasks/compositional.py,
run_compositional.py) concluded the local plasticity does NOT build compositional structure: a
NULL across PC depth, with held-out completion stuck at chance. A follow-up DIAGNOSIS found that
conclusion was largely a DEGENERATE-TASK ARTIFACT, for an information-theoretic reason:

  * f1 and f2 are INDEPENDENT latent factors (obs = concat(P1[f1], P2[f2])).
  * The completion target was the systematically-EXCLUDED f2 for each f1 (the held-out split
    holds out specific (f1,f2) pairs). Asking the model to recover that exact held-out f2 from
    the f1-part ALONE is information-theoretically IMPOSSIBLE: f1 carries zero information about
    f2. A backprop-MLP and a pure memorizer BOTH fail it too (see run_compositional.py: MLP
    ~0.067, memorizer 0.000). So the "completion" null measured the task, not the model.
  * Crucially, the SAME diagnosis found the trained latent LINEARLY DECODES BOTH f1 and f2 well
    above chance — i.e. the local rule DOES represent the two factors in x[top].

THE PRINCIPLED TEST (this file)
-------------------------------
Instead of an impossible f1->f2 completion, we test the right thing: is the latent FACTORIZED in
a way that GENERALIZES COMPOSITIONALLY? We:

  1. Train a bare CEREBRUM PCAreas hierarchy by the EXISTING local four-factor plasticity loop
     (benchmarks.tasks.compositional._train_pc) on a SUBSET of (f1,f2) combos.
  2. For every combo, settle the hierarchy NOISE-FREE (T=0) with the full obs clamped and read
     the top latent x[top].
  3. Fit a LINEAR readout (nearest-class-mean AND a small logistic/softmax classifier) on the
     latents of SEEN combos to predict f1, and (separately) to predict f2.
  4. EVALUATE decode accuracy on the HELD-OUT combos. This is genuine compositional
     generalization: can each factor be read off the latent for combinations never trained?

The readout is purely a MEASUREMENT PROBE (like the existing backprop_mlp COMPARATOR in
compositional.py). It is NOT part of CEREBRUM and uses no CEREBRUM machinery — the representation it
reads was learned entirely by the LOCAL rule. (Logistic GD lives only in this benchmark file,
clearly labelled; cerebrum/ is never touched and never does backprop.)

CONTROLS — so a positive result is not trivial
----------------------------------------------
The observation is a CONCAT of the two parts, so the factors are linearly present in the raw obs
already; any information-preserving map keeps them. To make this a fair test of the LEARNED
latent and not of the trivially-factorable input, we report THREE controls alongside CEREBRUM's
trained latent:

  * RAW obs            — decode factors directly from the raw observation (partly trivial: obs is
                         a concat, so the factor subspaces are axis-aligned and linearly read off).
  * RANDOM-PROJECTION  — decode from a fixed random linear projection of the obs to the SAME dim
                         as the latent. A random linear map preserves linear factor structure
                         (Johnson-Lindenstrauss), so this is the "no learning, just a generic
                         linear map of the same size" floor. If CEREBRUM's latent does no better
                         than this, the decoding is NOT evidence of LEARNED factorization.
  * UNTRAINED latent   — settle the SAME CEREBRUM architecture with its RANDOM init and NO plasticity,
                         then decode. The decisive learning control: any margin of TRAINED over
                         UNTRAINED is structure the LOCAL RULE actually built (vs the architecture's
                         inductive bias alone).

HONEST GOAL
-----------
Settle whether the local rule builds a FACTORIZED, compositionally-generalizing latent.
  * If held-out f1/f2 decode is HIGH (well above chance, CI-clean) -> the local rule DID build
    compositional structure and the earlier NULL was a degenerate-readout artifact (the honest
    CORRECTION). A clean margin of TRAINED over UNTRAINED / RANDOM-PROJECTION makes the case that
    LEARNING organized it, not just the factorable input or the architecture.
  * If held-out decode is at chance -> the earlier null stands.
Either way the verdict is printed from the actual numbers; nothing is engineered to win.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
import numpy as np

from cerebrum.config import CerebrumConfig
from cerebrum.pc_core import PCAreas
from cerebrum.rng import SeededRNG
from benchmarks.stats import mean_ci, fmt_ci
from benchmarks.tasks.compositional import CompositionalTask, _train_pc

# Fixed seed for the NOISE-FREE latent readout (same convention as compositional._EVAL_SEED):
# the settled latent is a pure function of the learned weights, not of settling noise.
_EVAL_SEED = 0xC0FFEE
# Fixed seed for the random-projection control (the same projection across all conditions/seeds
# would conflate; we seed it per-call so it is a generic random map, not a tuned one).
_RANDPROJ_SEED = 0x5EED


# ------------------------------------------------------------------------------------------
# Compositional train/holdout split (larger & coverage-guaranteed; not engineered for a result)
# ------------------------------------------------------------------------------------------

def make_split(A, B, frac_heldout=0.3, seed=0):
    """Hold out ~frac_heldout of the A*B combos at random, GUARANTEEING every factor value still
    appears in training (so each held-out factor is decodable in principle -> a fair compositional
    probe). Greedy: shuffle combos, add to held-out only while the remaining train set still
    covers all A f1-values and all B f2-values. Returns (train_combos, heldout_combos)."""
    rng = np.random.default_rng(seed)
    allc = [(f1, f2) for f1 in range(A) for f2 in range(B)]
    perm = allc[:]
    rng.shuffle(perm)
    held = set()
    n_target = int(frac_heldout * len(allc))
    for c in perm:
        if len(held) >= n_target:
            break
        cand = held | {c}
        train = [x for x in allc if x not in cand]
        if ({f for f, _ in train} == set(range(A)) and
                {f for _, f in train} == set(range(B))):
            held = cand
    train = [c for c in allc if c not in held]
    return train, sorted(held)


# ------------------------------------------------------------------------------------------
# Latent extraction (noise-free settle of the trained hierarchy)
# ------------------------------------------------------------------------------------------

def settle_top_latent(net, obs, steps, seed=_EVAL_SEED):
    """Clamp the FULL obs at the bottom, settle NOISE-FREE (T=0), return the top latent x[-1].
    Pure function of the learned weights (T=0 -> zero noise, deterministic)."""
    erng = SeededRNG(seed)
    net.x = [np.zeros_like(xl) for xl in net.x]
    for _ in range(steps):
        net.settle_step(erng, T=0.0, clamp_bottom=obs)
    return net.x[-1].copy()


# ------------------------------------------------------------------------------------------
# LINEAR-PROBE MEASUREMENT decoders (NOT part of CEREBRUM — pure benchmark-side measurement,
# exactly like the existing backprop_mlp COMPARATOR in compositional.py)
# ------------------------------------------------------------------------------------------

def _standardize(Xtr, Xte):
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ncm_decode_acc(Xtr, ytr, Xte, yte, n_cls):
    """Nearest-CLASS-MEAN linear probe: fit per-class means on the SEEN latents, classify
    HELD-OUT latents by nearest mean (standardized Euclidean). A measurement probe with no free
    nonlinearity — it can only succeed if the factor is LINEARLY organized in the latent."""
    if Xte.shape[0] == 0:
        return float("nan")
    Ztr, Zte = _standardize(Xtr, Xte)
    means = np.array([Ztr[ytr == c].mean(0) if np.any(ytr == c)
                      else np.full(Ztr.shape[1], 1e9) for c in range(n_cls)])
    d = np.sum((Zte[:, None, :] - means[None, :, :]) ** 2, axis=2)
    return float(np.mean(np.argmin(d, axis=1) == yte))


def logistic_decode_acc(Xtr, ytr, Xte, yte, n_cls, epochs=400, lr=0.5, l2=1e-3, seed=0):
    """Multinomial logistic (softmax) linear probe trained by plain gradient descent on the SEEN
    latents, evaluated on HELD-OUT latents. This GD is a MEASUREMENT PROBE only — it is NOT part
    of CEREBRUM (CEREBRUM never does backprop), exactly analogous to the backprop_mlp COMPARATOR that
    already lives in benchmarks/tasks/compositional.py. It reads, it does not teach."""
    if Xte.shape[0] == 0:
        return float("nan")
    Ztr, Zte = _standardize(Xtr, Xte)
    n, d = Ztr.shape
    rng = np.random.default_rng(seed)
    W = 0.01 * rng.standard_normal((n_cls, d))
    b = np.zeros(n_cls)
    Y = np.eye(n_cls)[ytr]
    for _ in range(epochs):
        logit = Ztr @ W.T + b
        logit -= logit.max(1, keepdims=True)
        p = np.exp(logit)
        p /= p.sum(1, keepdims=True)
        g = (p - Y) / n
        W -= lr * (g.T @ Ztr + l2 * W)
        b -= lr * g.sum(0)
    pred = np.argmax(Zte @ W.T + b, axis=1)
    return float(np.mean(pred == yte))


# ------------------------------------------------------------------------------------------
# One probe = train CEREBRUM, settle latents for all four conditions, decode f1 & f2 (held-out)
# ------------------------------------------------------------------------------------------

def factorization_probe(task, train, held, dims, passes=60, seed=0,
                        align_feedback=False, lam_kp=1e-2, decoder="both"):
    """Train a bare CEREBRUM hierarchy by the LOCAL rule on `train`, then linear-probe f1/f2 decoding
    on `held` from: the TRAINED latent, an UNTRAINED (random-init, no-plasticity) latent of the
    same architecture, the RAW obs, and a RANDOM-PROJECTION of the obs to the latent dim.

    Returns a flat dict of held-out decode accuracies. `decoder` in {"ncm","logreg","both"}:
    for "both" the returned values are the MEAN of the two probes (NCM and logistic), which is
    what run_one_seed reports per-kind separately; here "both" gives a single combined number per
    field for the smoke test and the per-seed summary."""
    A, B = task.A, task.B
    steps = 24  # settle long enough for the noise-free readout (>= the train-time settle count)
    f1tr = np.array([f for f, _ in train]); f2tr = np.array([f for _, f in train])
    f1te = np.array([f for f, _ in held]); f2te = np.array([f for _, f in held])

    # ---- train CEREBRUM by the LOCAL four-factor rule (no backprop) -------------------------
    cfg = CerebrumConfig(dims=dims, n_settle=12, seed=seed,
                      align_feedback=align_feedback, lam_kp=lam_kp)
    trained = _train_pc(task, cfg, passes=passes)
    untrained = PCAreas(cfg)  # same architecture / init, NO plasticity (learning control)

    def latents(net):
        Ltr = np.array([settle_top_latent(net, task.embed(*c), steps) for c in train])
        Lte = np.array([settle_top_latent(net, task.embed(*c), steps) for c in held])
        return Ltr, Lte

    Xtr, Xte = latents(trained)
    Utr, Ute = latents(untrained)
    Rtr = np.array([task.embed(*c) for c in train])
    Rte = np.array([task.embed(*c) for c in held])
    rp = np.random.default_rng(_RANDPROJ_SEED + seed).standard_normal((task.obs_dim, dims[-1]))
    Ptr = Rtr @ rp
    Pte = Rte @ rp

    def dec(Ztr, y_tr, Zte, y_te, n_cls):
        if decoder == "ncm":
            return ncm_decode_acc(Ztr, y_tr, Zte, y_te, n_cls)
        if decoder == "logreg":
            return logistic_decode_acc(Ztr, y_tr, Zte, y_te, n_cls, seed=seed)
        a = ncm_decode_acc(Ztr, y_tr, Zte, y_te, n_cls)
        b = logistic_decode_acc(Ztr, y_tr, Zte, y_te, n_cls, seed=seed)
        return 0.5 * (a + b)

    return {
        "trained_f1": dec(Xtr, f1tr, Xte, f1te, A),
        "trained_f2": dec(Xtr, f2tr, Xte, f2te, B),
        "untrained_f1": dec(Utr, f1tr, Ute, f1te, A),
        "untrained_f2": dec(Utr, f2tr, Ute, f2te, B),
        "raw_f1": dec(Rtr, f1tr, Rte, f1te, A),
        "raw_f2": dec(Rtr, f2tr, Rte, f2te, B),
        "randproj_f1": dec(Ptr, f1tr, Pte, f1te, A),
        "randproj_f2": dec(Ptr, f2tr, Pte, f2te, B),
        "n_train": len(train), "n_held": len(held),
    }


def run_one_seed(seed, A=6, B=6, part_dim=8, width=24, depth=3, frac_heldout=0.3,
                 passes=60, align_feedback=False, lam_kp=1e-2):
    """Run the probe once with BOTH decoder kinds reported separately (ncm, logreg)."""
    task = CompositionalTask(A=A, B=B, part_dim=part_dim, seed=seed)
    train, held = make_split(A, B, frac_heldout=frac_heldout, seed=1000 + seed)
    dims = tuple([task.obs_dim] + [width] * (depth - 1))
    out = {}
    for kind in ("ncm", "logreg"):
        out[kind] = factorization_probe(task, train, held, dims=dims, passes=passes,
                                        seed=seed, align_feedback=align_feedback,
                                        lam_kp=lam_kp, decoder=kind)
    out["n_train"] = out["ncm"]["n_train"]
    out["n_held"] = out["ncm"]["n_held"]
    return out


# ------------------------------------------------------------------------------------------
# Multi-seed sweep + reporting
# ------------------------------------------------------------------------------------------

def run_sweep(A=6, B=6, part_dim=8, width=24, depth=3, frac_heldout=0.3, passes=60,
              seeds=(0, 1, 2, 3, 4), align_feedback=False, lam_kp=1e-2):
    """Run the probe over seeds. Returns per-condition raw per-seed lists for f1 and f2 decode,
    for both decoder kinds, plus an averaged-over-factors summary."""
    fields = ["trained_f1", "trained_f2", "untrained_f1", "untrained_f2",
              "raw_f1", "raw_f2", "randproj_f1", "randproj_f2"]
    acc = {"ncm": {f: [] for f in fields}, "logreg": {f: [] for f in fields}}
    n_held = n_train = None
    for s in seeds:
        o = run_one_seed(s, A=A, B=B, part_dim=part_dim, width=width, depth=depth,
                         frac_heldout=frac_heldout, passes=passes,
                         align_feedback=align_feedback, lam_kp=lam_kp)
        n_held, n_train = o["n_held"], o["n_train"]
        for kind in ("ncm", "logreg"):
            for f in fields:
                acc[kind][f].append(o[kind][f])
    meta = dict(A=A, B=B, part_dim=part_dim, width=width, depth=depth,
                frac_heldout=frac_heldout, passes=passes, seeds=list(seeds),
                align_feedback=align_feedback, n_held=n_held, n_train=n_train,
                chance=1.0 / B)
    return {"acc": acc, "meta": meta}


def _avg_factor(acc_kind, cond):
    """Mean over (f1,f2) per seed for a condition, returned as a per-seed list (for CI)."""
    a1 = acc_kind[f"{cond}_f1"]
    a2 = acc_kind[f"{cond}_f2"]
    return [0.5 * (x + y) for x, y in zip(a1, a2)]


def _verdict(out):
    chance = out["meta"]["chance"]
    acc = out["acc"]
    # combine the two probe kinds (their mean) per seed for the headline decision
    def combo(cond):
        per_seed = []
        nseed = len(acc["ncm"][f"{cond}_f1"])
        for i in range(nseed):
            vals = [acc["ncm"][f"{cond}_f1"][i], acc["ncm"][f"{cond}_f2"][i],
                    acc["logreg"][f"{cond}_f1"][i], acc["logreg"][f"{cond}_f2"][i]]
            per_seed.append(float(np.mean(vals)))
        return per_seed
    tr = combo("trained"); un = combo("untrained"); rp = combo("randproj")
    mt, ht = mean_ci(tr); mu, hu = mean_ci(un); mr, hr = mean_ci(rp)
    lines = []
    above_chance = (mt - ht) > chance + 0.05
    beats_untrained = (mt - mu) > 0.05
    below_untrained = (mu - mt) > 0.05   # trained DEGRADED the architecture's own structure
    beats_randproj = (mt - mr) > 0.05
    def margin(m_trained, m_other):
        if (m_trained - m_other) > 0.05:
            return "TRAINED higher"
        if (m_other - m_trained) > 0.05:
            return "TRAINED LOWER"
        return "no clear margin"
    lines.append(f"trained latent factor-decode (held-out) = {mt:.3f} +/- {ht:.3f}  "
                 f"(chance {chance:.3f})")
    lines.append(f"  vs untrained latent  {mu:.3f} +/- {hu:.3f}   ({margin(mt, mu)})")
    lines.append(f"  vs random-projection {mr:.3f} +/- {hr:.3f}   ({margin(mt, mr)})")
    if below_untrained:
        lines.append(
            "VERDICT: LEARNING DEGRADES THE LATENT — held-out factor decode is clearly BELOW the "
            "UNTRAINED same-architecture latent, so this configuration of the local rule actively "
            "WORSENS the linearly-decodable factored structure the architecture starts with. "
            f"(Trained {mt:.3f} < untrained {mu:.3f}.) This is an honest negative for this "
            "configuration; report it as such (do not read it as 'factorization confirmed').")
        return "\n".join(lines)
    if not above_chance:
        lines.append("VERDICT: NULL STANDS — even with the principled linear probe, the trained "
                     "latent does NOT decode the factors above chance on held-out combos.")
        return "\n".join(lines)
    # above chance -> the latent IS factorized for unseen combos
    head = ("VERDICT: CORRECTS THE NULL — the trained latent DOES carry a compositionally-"
            "generalizing factored code: f1 and f2 are LINEARLY decodable on HELD-OUT combos "
            "well above chance. The earlier f1->f2 'completion' null was a degenerate "
            "(information-theoretically unsolvable) readout, not absence of factorization.")
    if beats_untrained:
        head += (" The margin over the UNTRAINED same-architecture latent shows the LOCAL RULE "
                 "actively organized this structure (not just the architecture's bias).")
    if beats_randproj:
        head += (" It also exceeds a random-projection of the obs of the same dim, so the decode "
                 "is not merely inherited from the trivially-factorable concat input.")
    elif not beats_randproj:
        head += (" HONEST CAVEAT: it does NOT clearly exceed a random-projection of the obs (the "
                 "concat input is already linearly factorable), so part of the decodability is "
                 "inherited from the input; the load-bearing learned evidence is the margin over "
                 "the UNTRAINED latent.")
    lines.append(head)
    return "\n".join(lines)


def _print_block(out, title):
    m = out["meta"]
    acc = out["acc"]
    print(title)
    print(f"A={m['A']} f1 x B={m['B']} f2; part_dim={m['part_dim']} (obs_dim={2*m['part_dim']}); "
          f"dims=(obs,{','.join([str(m['width'])]*(m['depth']-1))}); passes={m['passes']}; "
          f"seeds={len(m['seeds'])}; chance=1/B={m['chance']:.3f}")
    print(f"train combos={m['n_train']}, held-out combos={m['n_held']} per seed "
          f"(align_feedback={m['align_feedback']})")
    print()
    print(f"{'condition':>20}  {'f1 decode (held-out)':>24}  {'f2 decode (held-out)':>24}  "
          f"{'factor-avg':>20}")
    conds = [("TRAINED latent", "trained"), ("UNTRAINED latent", "untrained"),
             ("RAW obs", "raw"), ("RANDOM-PROJECTION", "randproj")]
    for kind in ("ncm", "logreg"):
        label = "nearest-class-mean probe" if kind == "ncm" else "logistic (GD) probe"
        print(f"-- {label} --")
        a = acc[kind]
        for name, cond in conds:
            avg = _avg_factor(a, cond)
            print(f"{name:>20}  {fmt_ci(a[f'{cond}_f1']):>24}  {fmt_ci(a[f'{cond}_f2']):>24}  "
                  f"{fmt_ci(avg):>20}")
        print()


if __name__ == "__main__":
    print("=" * 96)
    print("PRINCIPLED FACTORIZATION PROBE — does the LOCAL rule build a compositionally-")
    print("generalizing FACTORED latent? (corrects the earlier degenerate f1->f2 completion null)")
    print("=" * 96)
    print("Linear readouts are MEASUREMENT probes only (like the existing backprop_mlp comparator);")
    print("CEREBRUM itself does NO backprop and is unmodified. The latent was learned by the LOCAL rule.")
    print()

    base = run_sweep()
    _print_block(base, ">>> align_feedback = OFF (default local rule)")
    print(_verdict(base))
    print()

    print("=" * 96)
    kp = run_sweep(align_feedback=True)
    _print_block(kp, ">>> align_feedback = ON (Kolen-Pollack B->W^T alignment; does it change factorization?)")
    print(_verdict(kp))
