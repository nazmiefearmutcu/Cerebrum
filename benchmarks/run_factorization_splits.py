"""C2-HardSplits — does the LOCAL rule generalize SYSTEMATICALLY, or only INTERPOLATE?

WHAT THIS EXTENDS
-----------------
benchmarks/run_factorization.py (§g) and run_factorization_multi.py (§g2, C1-MoreFactors) showed
that the LOCAL four-factor plasticity builds a compositionally-generalizing FACTORED latent under a
RANDOM held-out split of the combo grid: each factor is linearly decodable off the trained top
latent x[top] on held-out combos, above the untrained same-arch latent (the LOAD-BEARING learned
margin) and — at small cardinality — above a random-projection of the obs.

But a RANDOM held-out split tests only INTERPOLATION: every held-out factor value appears in MANY
training combos, so the latent need only interpolate within a densely-sampled grid. The deeper
compositional question is SYSTEMATICITY / PRODUCTIVITY: can a factor value be read off / used in
NEW contexts it was barely (or never-as-a-pair) trained in? This probe builds HARDER hold-out
structure and HONESTLY maps where the local rule's factorization HOLDS vs BREAKS:

  (a) LEAVE-A-VALUE-IN-FEW-CONTEXTS — a particular factor value appears in training in only
      n_contexts in {1, 2} combinations; the held-out set is the OTHER contexts of that value.
      Test: is that value still linearly decodable off the latent for the NEW contexts? This asks
      whether the factor SUBSPACE generalizes from SPARSE context exposure (systematic) or whether
      the latent has only bound that value to the specific co-factors it was seen with (interpolate).

  (b) ROW hold-out (PRODUCTIVITY) — hold out a whole structured REGION of the grid: target_factor
      fixed to target_value, crossed with a block of the OTHER factor's values (a "row"). The
      target value still appears in training in the NON-held columns of its row (decodable in
      principle), but the held cells are a CONTIGUOUS structured region, not random scatter. Test:
      decode the held factor in that structured-out region.

FAIRNESS / DECODABILITY-IN-PRINCIPLE
------------------------------------
Each split builder enforces the invariants that make a null INFORMATIVE rather than a
trivially-unsolvable artifact: (i) every factor VALUE still appears somewhere in training (so each
factor is decodable in principle), (ii) for few-context, the target value is still SEEN in training
(sparsely), (iii) for row, the target value is seen in its non-held columns and the held co-factor
values are seen via other rows. We never engineer the answer to be readable from the input — the
controls below catch that.

CONTROLS (so a "win" is attributable to the LEARNED latent, not architecture / input)
-------------------------------------------------------------------------------------
  * UNTRAINED latent  — SAME architecture, random init, NO plasticity, settled. The LOAD-BEARING
                        learned control: any margin of TRAINED over UNTRAINED is structure the
                        LOCAL RULE built beyond the architecture's init bias.
  * RANDOM-PROJECTION — a fixed random linear map of the obs to the latent dim (Johnson-Lindenstrauss
                        preserves linear factor structure). The "no learning, generic same-dim
                        linear map" floor on the trivially-factorable concat input.
  (RAW obs is also computed for the per-block table.)

HEADLINE PROBE = NEAREST-CLASS-MEAN. As established in C1-MoreFactors, the logistic-GD readout is
OVER-POWERED at this train-set size and saturates ALL conditions to ~1.0, so it cannot discriminate
learned structure from the input's trivial linear factorability. The NCM probe has no free
nonlinearity, so its margins are real signal. Both are reported in the table; NCM drives the verdict.

THE TARGET METRIC per split
---------------------------
  * few_context / row : decode of the TARGET factor on the held-out (new-context / structured-out)
                        combos — that is exactly the systematic-generalization question.
  * random            : the same target-factor decode on the random held-out (the INTERPOLATION
                        reference). We use the SAME target factor across all three split kinds so the
                        comparison is apples-to-apples (one factor, three hold-out structures).

VERDICT (printed from the actual numbers; nothing engineered to win)
--------------------------------------------------------------------
Factor-averaged-over-seeds, 95% CIs. For each split we ask the SAME three questions as §g2:
above chance? margin over UNTRAINED (load-bearing)? margin over RANDOM-PROJECTION (strongest)?
Then the headline verdict compares the HARD splits to the RANDOM split:
  * SYSTEMATIC — the hard-split target decode HOLDS (above chance, clear margin over untrained),
                 i.e. not materially worse than the random-split interpolation reference.
  * INTERPOLATE-ONLY — the random split holds but the hard split(s) collapse toward untrained /
                 chance: the latent only interpolates within densely-sampled combos and does NOT
                 generalize a factor value to new / structured-out contexts.
  * MIXED — holds for one hard split but breaks for the other (mapped explicitly).

grail/ is NEVER touched and does NO backprop. The linear readouts are MEASUREMENT probes only
(shared with run_factorization.py). The latent was learned entirely by the LOCAL rule.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
import numpy as np

from grail.config import GRAILConfig
from grail.pc_core import PCAreas
from benchmarks.stats import mean_ci, fmt_ci
from benchmarks.run_factorization import ncm_decode_acc, logistic_decode_acc
from benchmarks.run_factorization_multi import (
    MultiFactorTask, settle_top_latent_multi, _train_pc_multi,
)

# Fixed seed offset for the random-projection control (seeded per-call so it is a GENERIC random
# map of the obs, not one tuned to win) — same convention as run_factorization_multi.
_RANDPROJ_SEED = 0x5EED

SPLIT_KINDS = ("random", "few_context", "row")


# ==========================================================================================
# SPLIT BUILDERS — each returns (train_combos, heldout_combos, meta)
#   meta carries: target_factor, target_value (the factor whose held-out decode is the headline),
#   and a 'kind' tag. All builders enforce decodability-in-principle invariants (see module docstr).
# ==========================================================================================

def _full_grid(cards):
    if len(cards) == 2:
        return [(a, b) for a in range(cards[0]) for b in range(cards[1])]
    # general K (the hard splits are defined on 2 named factors; extra factors vary freely)
    import itertools
    return [tuple(c) for c in itertools.product(*[range(c) for c in cards])]


def make_random_split(cards, frac_heldout=0.3, seed=0, target_factor=0, target_value=None):
    """RANDOM hold-out (the INTERPOLATION reference / control). Hold out ~frac_heldout of the full
    grid at random while GUARANTEEING every factor value still appears in training. target_value is
    irrelevant here (random decode averages over the target factor's values); kept for a uniform
    signature. Returns (train, held, meta)."""
    cards = tuple(int(c) for c in cards)
    K = len(cards)
    rng = np.random.default_rng(seed)
    allc = _full_grid(cards)
    perm = allc[:]
    rng.shuffle(perm)
    held = set()
    n_target = int(frac_heldout * len(allc))
    for c in perm:
        if len(held) >= n_target:
            break
        cand = held | {c}
        train = [x for x in allc if x not in cand]
        if all({t[k] for t in train} == set(range(cards[k])) for k in range(K)):
            held = cand
    train = [c for c in allc if c not in held]
    meta = {"kind": "random", "target_factor": target_factor, "target_value": None,
            "cards": cards}
    return train, sorted(held), meta


def make_few_context_split(cards, n_contexts=2, target_factor=0, target_value=None, seed=0):
    """LEAVE-A-VALUE-IN-FEW-CONTEXTS (globally-sparse TARGET factor). EVERY value of the target
    factor appears in training in EXACTLY `n_contexts` combos; ALL OTHER combos are HELD OUT (the
    NEW contexts to which each target value must generalize). Returns (train, held, meta).

    WHY GLOBAL (every value), not a single singled-out value: a single target value seen in 2
    combos yields only a handful of held-out cells, so per-seed decode is a coarse 0/4..4/4 fraction
    and seed variance swamps any signal (statistically unsound CIs). Making EVERY target value
    sparse gives a LARGE held-out set (card*(card-n_contexts) cells), so the held-out decode is a
    stable estimate and the CIs are meaningful. The compositional question is identical and even
    sharper: each target value is bound to only n_contexts co-factor contexts in training; can the
    latent decode it for the MANY contexts it was never paired with? (systematic) or does it scatter
    new-context instances by co-factor? (interpolate-only).

    Decodability in principle: each target value IS seen in training (n_contexts combos) so its NCM
    class-mean is estimable; and we GUARANTEE every co-factor value also appears in training (greedy
    coverage repair below), so no factor is unseen. target_value is ignored (all values are sparse);
    kept in the signature for a uniform builder API."""
    cards = tuple(int(c) for c in cards)
    K = len(cards)
    tf = int(target_factor)
    rng = np.random.default_rng(seed)
    allc = _full_grid(cards)
    # for each target-factor value, KEEP n_contexts of its combos in training (chosen at random)
    keep = set()
    by_tv = {}
    for c in allc:
        by_tv.setdefault(c[tf], []).append(c)
    for tv, combos in by_tv.items():
        perm = combos[:]
        rng.shuffle(perm)
        keep.update(perm[:n_contexts])
    # coverage repair: ensure EVERY value of every co-factor appears in train; if a co-factor value
    # is absent, pull one held combo carrying it into train (keeps the split fair / decodable).
    def covered(train_set, k):
        return {t[k] for t in train_set}
    for k in range(K):
        if k == tf:
            continue
        missing = set(range(cards[k])) - covered(keep, k)
        for v in missing:
            cand = [c for c in allc if c not in keep and c[k] == v]
            if cand:
                keep.add(cand[int(rng.integers(len(cand)))])
    train = sorted(keep)
    held = sorted([c for c in allc if c not in keep])
    meta = {"kind": "few_context", "target_factor": tf, "target_value": None,
            "n_contexts": n_contexts, "cards": cards}
    return train, held, meta


def make_row_split(cards, target_factor=0, target_value=0, n_held_in_row=None,
                   block_rows=None, seed=0):
    """ROW/BLOCK hold-out (PRODUCTIVITY). Hold out a structured RECTANGULAR REGION of the grid: a
    BLOCK of `block_rows` target-factor values crossed with a contiguous set of the OTHER factor's
    values. This is the classic productivity test: a whole structured region (rows x columns), not
    random scatter. Returns (train, held, meta).

    Decodability in principle: each held target-factor value still appears in training in its
    NON-held columns (we cap held columns at card_other - 1, leaving >=1 column per held row), and
    each held co-factor value still appears in training via the NON-held rows (we cap block_rows at
    card_tf - 1). So every factor value is seen in training; the held cells are a structured block
    the model was never trained on as pairs.

    block_rows defaults to ~half the target rows; n_held_in_row defaults to ~60% of the co-factor
    values. Together they make the held-out region a sizeable structured rectangle (so per-seed
    decode is a stable estimate -> meaningful CIs), unlike a single thin row. target_value seeds
    WHICH rows form the block (rows target_value..target_value+block_rows-1 mod card_tf)."""
    cards = tuple(int(c) for c in cards)
    K = len(cards)
    tf = int(target_factor)
    other = 1 - tf if K == 2 else (tf + 1) % K  # the co-factor whose values the block spans
    card_tf = cards[tf]
    card_other = cards[other]
    if block_rows is None:
        block_rows = max(1, card_tf // 2)
    block_rows = min(block_rows, card_tf - 1)            # keep >=1 row (target value) fully trained
    if n_held_in_row is None:
        n_held_in_row = max(1, int(round(0.6 * card_other)))
    n_held_in_row = min(n_held_in_row, card_other - 1)   # keep >=1 column per held row in training
    rng = np.random.default_rng(seed)
    base = int(target_value)
    held_rows = {(base + i) % card_tf for i in range(block_rows)}
    other_vals = list(range(card_other))
    rng.shuffle(other_vals)
    held_cols = set(other_vals[:n_held_in_row])
    allc = _full_grid(cards)
    held = sorted([c for c in allc if c[tf] in held_rows and c[other] in held_cols])
    train = [c for c in allc if c not in set(held)]
    meta = {"kind": "row", "target_factor": tf, "target_value": base, "other_factor": other,
            "n_held_in_row": n_held_in_row, "block_rows": block_rows,
            "held_rows": sorted(held_rows), "cards": cards}
    return train, held, meta


# ==========================================================================================
# ONE PROBE: train GRAIL by the LOCAL rule, settle latents, decode the TARGET factor (held-out)
# ==========================================================================================

def split_probe(task, train, held, dims, passes=60, seed=0, target_factor=0,
                align_feedback=False, lam_kp=1e-2, decoder="ncm"):
    """Train a bare GRAIL hierarchy by the LOCAL rule on `train`, then linear-probe the TARGET
    factor's held-out decode off the TRAINED latent, an UNTRAINED (random-init, no-plasticity)
    latent, the RAW obs, and a RANDOM-PROJECTION of the obs. Returns a flat dict with
    {cond}_target for each condition plus the per-other-factor decode (for the table) + counts.

    `decoder` in {"ncm","logreg","both"}: "both" returns the MEAN of the two probes per field."""
    K = task.K
    steps = 24  # noise-free settle long enough for the readout (>= train-time settle count)
    ytr = np.array([list(c) for c in train])   # (n_train, K)
    yte = np.array([list(c) for c in held])    # (n_held,  K)

    cfg = GRAILConfig(dims=dims, n_settle=12, seed=seed,
                      align_feedback=align_feedback, lam_kp=lam_kp)
    trained = _train_pc_multi(task, train, cfg, passes=passes)
    untrained = PCAreas(cfg)  # same architecture / init, NO plasticity (learning control)

    def latents(net):
        Ltr = np.array([settle_top_latent_multi(net, task.embed(*c), steps) for c in train])
        Lte = np.array([settle_top_latent_multi(net, task.embed(*c), steps) for c in held])
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

    tf = int(target_factor)
    out = {"n_train": len(train), "n_held": len(held)}
    conds = {"trained": (Xtr, Xte), "untrained": (Utr, Ute),
             "raw": (Rtr, Rte), "randproj": (Ptr, Pte)}
    for cname, (Ztr, Zte) in conds.items():
        # headline: TARGET factor decode on held-out
        out[f"{cname}_target"] = dec(Ztr, ytr[:, tf], Zte, yte[:, tf], task.cards[tf])
        # also the co-factor decode (for the table; the held-out co-factor values are densely seen)
        for k in range(K):
            if k == tf:
                continue
            out[f"{cname}_f{k}"] = dec(Ztr, ytr[:, k], Zte, yte[:, k], task.cards[k])
    return out


def run_one_seed_splits(seed, cards=(8, 8), part_dim=8, width=24, depth=3, passes=60,
                        target_factor=0, target_value=2, n_contexts=2, frac_heldout=0.3,
                        n_held_in_row=None, align_feedback=False, lam_kp=1e-2):
    """Run all THREE split kinds once on the SAME task & SAME target factor, with BOTH decoder kinds
    reported separately (ncm, logreg). Returns out[kind][probe][...] + per-kind chance / counts.

    The target factor's chance is 1/cards[target_factor]; we report that as the headline chance for
    every split (the target metric is target-factor decode in all three)."""
    task = MultiFactorTask(cards=cards, part_dim=part_dim, seed=seed)
    tf = int(target_factor)
    chance = 1.0 / cards[tf]
    splits = {
        "random": make_random_split(cards, frac_heldout=frac_heldout, seed=2000 + seed,
                                     target_factor=tf),
        "few_context": make_few_context_split(cards, n_contexts=n_contexts, target_factor=tf,
                                               target_value=target_value, seed=3000 + seed),
        "row": make_row_split(cards, target_factor=tf, target_value=target_value,
                              n_held_in_row=n_held_in_row, seed=4000 + seed),
    }
    dims = tuple([task.obs_dim] + [width] * (depth - 1))
    out = {}
    for kind, (train, held, meta) in splits.items():
        rec = {"chance": chance, "meta": meta, "n_train": len(train), "n_held": len(held)}
        for probe in ("ncm", "logreg"):
            rec[probe] = split_probe(task, train, held, dims=dims, passes=passes, seed=seed,
                                     target_factor=tf, align_feedback=align_feedback,
                                     lam_kp=lam_kp, decoder=probe)
        out[kind] = rec
    return out


# ==========================================================================================
# MULTI-SEED SWEEP + VERDICT
# ==========================================================================================

def run_sweep_splits(cards=(8, 8), part_dim=8, width=24, depth=3, passes=60,
                     seeds=(0, 1, 2, 3, 4), target_factor=0, target_value=2, n_contexts=2,
                     frac_heldout=0.3, n_held_in_row=None, align_feedback=False, lam_kp=1e-2):
    """Run all three split kinds over seeds. Returns per-kind, per-probe, per-condition per-seed
    target-decode lists + metadata. The headline per (kind,probe,cond) is the TARGET-factor held-out
    decode per seed (already a single number)."""
    tf = int(target_factor)
    chance = 1.0 / cards[tf]
    conds = ("trained", "untrained", "raw", "randproj")
    acc = {kind: {probe: {f"{c}_target": [] for c in conds} for probe in ("ncm", "logreg")}
           for kind in SPLIT_KINDS}
    counts = {kind: {"n_train": [], "n_held": []} for kind in SPLIT_KINDS}
    for s in seeds:
        o = run_one_seed_splits(s, cards=cards, part_dim=part_dim, width=width, depth=depth,
                                passes=passes, target_factor=tf, target_value=target_value,
                                n_contexts=n_contexts, frac_heldout=frac_heldout,
                                n_held_in_row=n_held_in_row, align_feedback=align_feedback,
                                lam_kp=lam_kp)
        for kind in SPLIT_KINDS:
            counts[kind]["n_train"].append(o[kind]["n_train"])
            counts[kind]["n_held"].append(o[kind]["n_held"])
            for probe in ("ncm", "logreg"):
                for c in conds:
                    acc[kind][probe][f"{c}_target"].append(o[kind][probe][f"{c}_target"])
    meta = dict(cards=cards, part_dim=part_dim, width=width, depth=depth, passes=passes,
                seeds=list(seeds), target_factor=tf, target_value=target_value,
                n_contexts=n_contexts, frac_heldout=frac_heldout, align_feedback=align_feedback,
                chance=chance, counts=counts)
    return {"acc": acc, "meta": meta}


def _classify_split(acc_kind, chance, probe="ncm"):
    """Honest verdict for ONE split kind. The LOAD-BEARING signal (as established in C1-MoreFactors)
    is the PAIRED learned margin trained - untrained WITHIN this split: both conditions suffer the
    SAME sparse-class-mean handicap that a hard split imposes (a value seen in few contexts yields a
    noisy NCM mean even for an oracle), so their per-seed DIFFERENCE isolates the structure the
    LOCAL RULE built from the difficulty of the readout itself. We compute that margin PAIRED
    per-seed (tighter + correct: it is a within-seed contrast, not a difference of two independent
    CIs). We also report the random-projection margin (the stronger learned-beyond-input claim).

    Labels (paired-margin centric):
      HOLDS   — above chance AND the PAIRED learned margin /init is CI-clean > 0 AND clear margin
                over random-proj.
      PARTIAL — above chance AND PAIRED learned margin /init CI-clean > 0, but NOT clearly over
                random-proj (decode partly inherited from the trivially-factorable concat input).
      BREAKS  — paired learned margin /init not CI-clean > 0 (collapsed to the architecture's init
                bias), or below init, or not above chance."""
    tr = np.array(acc_kind[probe]["trained_target"], float)
    un = np.array(acc_kind[probe]["untrained_target"], float)
    rp = np.array(acc_kind[probe]["randproj_target"], float)
    mt, ht = mean_ci(tr); mu, hu = mean_ci(un); mr, hr = mean_ci(rp)
    # PAIRED per-seed learned margins (within-seed contrasts)
    md_init, hd_init = mean_ci(tr - un)        # trained - untrained, paired
    md_rp, hd_rp = mean_ci(tr - rp)            # trained - randproj, paired
    above_chance = (mt - ht) > chance + 0.05
    margin_init_clean = (md_init - hd_init) > 0.0    # CI-clean positive learned margin over init
    margin_rp_clean = (md_rp - hd_rp) > 0.0          # CI-clean positive over the input floor
    below_init = (md_init + hd_init) < 0.0           # CI-clean BELOW init (learning degraded it)
    if below_init or not above_chance or not margin_init_clean:
        label = "BREAKS"
    elif margin_rp_clean:
        label = "HOLDS"
    else:
        label = "PARTIAL"
    return label, (mt, ht, mu, hu, mr, hr, md_init, hd_init, md_rp, hd_rp)


def _print_block(out, title):
    m = out["meta"]
    acc = out["acc"]
    print(title)
    tf = m["target_factor"]
    print(f"cards={m['cards']}; target factor=f{tf} (value={m['target_value']} for the hard "
          f"splits); part_dim={m['part_dim']} (obs_dim={len(m['cards'])*m['part_dim']}); "
          f"dims=(obs,{','.join([str(m['width'])]*(m['depth']-1))}); passes={m['passes']}; "
          f"seeds={len(m['seeds'])}; chance=1/card={m['chance']:.3f}; "
          f"align_feedback={m['align_feedback']}")
    print(f"few_context: target value seen in only n_contexts={m['n_contexts']} train combos; "
          f"row: ~60% of the target row held out as a structured region")
    print()
    for kind in SPLIT_KINDS:
        cnt = m["counts"][kind]
        ntr = int(np.mean(cnt["n_train"])); nhe = int(np.mean(cnt["n_held"]))
        print(f"  [{kind}]  (avg train={ntr}, held-out={nhe})")
        for probe in ("ncm", "logreg"):
            label = "NCM" if probe == "ncm" else "logreg(GD)"
            a = acc[kind][probe]
            print(f"    {label:>10}: trained {fmt_ci(a['trained_target'])}  | "
                  f"untrained {fmt_ci(a['untrained_target'])}  | "
                  f"randproj {fmt_ci(a['randproj_target'])}  | "
                  f"raw {fmt_ci(a['raw_target'])}")
        print()


def verdict(out, probe="ncm"):
    """Render the HOLDS/PARTIAL/BREAKS label per split and the headline SYSTEMATIC-vs-INTERPOLATE
    verdict. Printed from numbers.

    The headline is decided on the LOAD-BEARING PAIRED LEARNED MARGIN (trained - untrained, within
    seed) — NOT on the absolute decode level. WHY: a hard split (sparse-context / structured row)
    raises the difficulty of the linear READOUT itself (a target value seen in only 2 contexts gives
    a noisy NCM class-mean even for a perfectly-factorized oracle), so the ABSOLUTE decode is
    EXPECTED to drop under the hard splits regardless of the rule. The fair, structure-isolating
    contrast is trained vs untrained UNDER THE SAME SPLIT: both suffer the identical readout
    handicap, so a surviving paired margin is structure the LOCAL RULE built that GENERALIZES to the
    new / structured-out contexts. If that paired margin survives (CI-clean > 0) under the hard
    splits, the rule is SYSTEMATIC; if it collapses to 0 (or below) only under the hard splits while
    holding for random, it merely INTERPOLATES."""
    chance = out["meta"]["chance"]
    acc = out["acc"]
    labels = {}
    stats = {}
    lines = []
    for kind in SPLIT_KINDS:
        lab, st = _classify_split(acc[kind], chance, probe=probe)
        labels[kind] = lab
        stats[kind] = st
        mt, ht, mu, hu, mr, hr, md_i, hd_i, md_r, hd_r = st
        lines.append(f"  [{kind:>12}] trained={mt:.3f}+/-{ht:.3f}  "
                     f"(paired /init {md_i:+.3f}+/-{hd_i:.3f}, paired /input {md_r:+.3f}+/-{hd_r:.3f}, "
                     f"chance {chance:.3f})  -> {lab}")
    head_lines = []
    hard = ("few_context", "row")
    # SYSTEMATIC iff the random reference holds AND the PAIRED learned margin /init is CI-clean > 0
    # under BOTH hard splits (i.e. neither hard split is labelled BREAKS).
    rnd_ok = labels["random"] in ("HOLDS", "PARTIAL")
    hard_margin_survives = all(labels[k] in ("HOLDS", "PARTIAL") for k in hard)
    hard_all_break = all(labels[k] == "BREAKS" for k in hard)
    if not rnd_ok:
        # the random reference is below CI-clean here (typically UNDERPOWERED: small held-out set /
        # higher per-seed variance at this cardinality), so the COMPARATIVE 'survives vs random'
        # claim cannot be made cleanly. Report it precisely, and note whether the HARD splits' own
        # learned margins are individually clean (they often are — sparse context degrades the
        # untrained baseline more, widening the margin).
        hard_clean = [k for k in hard if labels[k] in ("HOLDS", "PARTIAL")]
        extra = ("" if not hard_clean else
                 f" NOTE: the hard split(s) {hard_clean} DO show a CI-clean positive learned margin "
                 f"/init here (the sparse-context readout degrades the UNTRAINED baseline at least as "
                 f"much as the trained latent), so the local rule's factorization is not failing under "
                 f"the hard structure — the inconclusiveness is in the underpowered RANDOM baseline.")
        head = ("VERDICT: INCONCLUSIVE (underpowered random reference) — the RANDOM (interpolation) "
                "reference's paired learned margin /init is not CI-clean above 0 at this config, so "
                "the COMPARATIVE 'survives the hard split relative to random' claim cannot be made "
                "cleanly." + extra)
    elif hard_margin_survives:
        head = ("VERDICT: SYSTEMATIC (on the load-bearing learned margin) — the local rule's "
                "factorization generalizes BEYOND mere interpolation. The PAIRED learned margin over "
                "the untrained same-arch latent (the structure the LOCAL RULE built) stays CI-clean "
                "POSITIVE under BOTH hard splits: when each target value is seen in only n_contexts "
                "training combos (leave-a-value-in-few-contexts) AND when a whole structured "
                "rectangular block of the grid is held out (productivity). The ABSOLUTE held-out "
                "decode does drop under the sparse-context split (the readout itself is harder when a "
                "class is seen in few contexts — an oracle drops there too), but the trained latent "
                "keeps a real learned edge over init in exactly those hard contexts -> the factored "
                "SUBSPACE generalizes across contexts, it is not bound to the co-factors a value was "
                "paired with.")
    elif hard_all_break:
        head = ("VERDICT: INTERPOLATE-ONLY — the random-split positive does NOT survive systematic "
                "hold-out. The PAIRED learned margin /init holds for the RANDOM split (dense "
                "interpolation) but COLLAPSES to ~0 (or below) under BOTH hard splits. The local rule "
                "INTERPOLATES within a densely-sampled grid but does not systematically generalize a "
                "factor value to new / structured-out contexts.")
    else:
        head = ("VERDICT: MIXED — the paired learned margin /init survives one hard split but "
                "collapses for the other (see per-split labels). The local rule's factorization is "
                "partly systematic and partly interpolative; the boundary is mapped per split above.")
    head_lines.append(head)
    return "\n".join(lines + [""] + head_lines), labels, stats


def run_grid(seeds=tuple(range(12)), passes=60, width=24, depth=3, part_dim=8):
    """Sweep a couple of cardinalities and print, for each, the per-split HOLDS/PARTIAL/BREAKS
    labels + the SYSTEMATIC-vs-INTERPOLATE headline. We use 2-factor tasks (the cleanest setting for
    a row/column structured hold-out) at cardinality 6 and 8 (matching C1-MoreFactors' hard end).
    chance = 1/card of the target factor."""
    configs = [
        ("card=6", (6, 6), 1),   # target_value index chosen < card
        ("card=8", (8, 8), 2),
    ]
    summary = []
    for name, cards, tval in configs:
        print("=" * 100)
        out = run_sweep_splits(cards=cards, part_dim=part_dim, width=width, depth=depth,
                               passes=passes, seeds=seeds, target_value=tval, n_contexts=2)
        _print_block(out, f">>> {name}  (2-factor; target value seen in 2 contexts for few_context; "
                           f"~60% row held out; align_feedback OFF)")
        txt, labels, stats = verdict(out)
        print(txt)
        print()
        summary.append((name, labels, stats, out["meta"]["chance"]))
    print("=" * 100)
    print("C2-HARDSPLITS MAP -- NCM headline probe, TARGET-factor held-out decode by split structure.")
    print("'paired /init' = within-seed (trained - untrained) +/- 95% CI = the LOAD-BEARING learned")
    print("margin; 'paired /input' = within-seed (trained - randproj).")
    print(f"  {'config':>8}  {'split':>12}  {'trained':>14}  {'untrained':>9}  "
          f"{'paired /init':>16}  {'paired /input':>16}  {'chance':>7}  verdict")
    for name, labels, stats, ch in summary:
        for kind in SPLIT_KINDS:
            mt, ht, mu, hu, mr, hr, md_i, hd_i, md_r, hd_r = stats[kind]
            print(f"  {name:>8}  {kind:>12}  {mt:>7.3f}+/-{ht:<5.3f}  {mu:>9.3f}  "
                  f"{md_i:>+8.3f}+/-{hd_i:<5.3f}  {md_r:>+8.3f}+/-{hd_r:<5.3f}  {ch:>7.3f}  "
                  f"{labels[kind]}")
    print()
    print("READING THE MAP: compare the RANDOM (interpolation) row to the few_context / row")
    print("(systematic) rows WITHIN each config. The ABSOLUTE 'trained' decode is EXPECTED to drop")
    print("under few_context (a value seen in few contexts -> noisier class-mean even for an oracle).")
    print("The honest systematicity signal is the PAIRED learned margin /init (trained vs untrained")
    print("UNDER THE SAME SPLIT): if it stays CI-clean POSITIVE under the hard splits, the local rule")
    print("generalizes the factor SYSTEMATICALLY; if it collapses to ~0 only under the hard splits,")
    print("it merely INTERPOLATES within densely-sampled combos.")
    return summary


if __name__ == "__main__":
    print("=" * 100)
    print("C2-HardSplits — does the LOCAL rule generalize SYSTEMATICALLY (decode holds under hard")
    print("hold-out structure) or only INTERPOLATE (holds for random split, breaks for systematic)?")
    print("=" * 100)
    print("Linear readouts are MEASUREMENT probes only (shared with run_factorization.py); GRAIL")
    print("itself does NO backprop and is unmodified. The latent was learned by the LOCAL rule.")
    print()
    run_grid()
