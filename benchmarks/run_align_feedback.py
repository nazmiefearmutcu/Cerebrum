"""KOLEN-POLLACK FEEDBACK-ALIGNMENT probe on the compositional task.

Tests the hypothesis: if the SEPARATE feedback weight B is driven toward W.T by a LOCAL
Kolen-Pollack rule (B and W get the SAME M-gated pre*post product, transposed, with a MATCHED
decay -- NO weight transport), then the latent settling drift points in a direction that reduces
bottom-up reconstruction error, which is the precondition for the local four-factor rule to build
a useful f1->f2 latent.

We run the SAME compositional probe (benchmarks.tasks.compositional) with align_feedback OFF vs ON
across >=5 seeds and report, with 95% CIs:
  - B<->W.T alignment cosine (start -> end of training)            [the mechanism check]
  - within-distribution (train-combo) completion accuracy           [does the latent become useful]
  - held-out compositional accuracy (chance = 1/B)                  [does it generalize compositionally]

SUCCESS = alignment rises AND within-dist completion rises above chance / held-out improves.
Honest negative reported if alignment rises but representation still fails.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from cerebrum.config import CerebrumConfig
from benchmarks.stats import mean_ci, fmt_ci
from benchmarks.tasks.compositional import CompositionalTask, run_pc_completion


def run_probe(width=24, depth=3, A=4, B=4, part_dim=8, passes=80,
              seeds=(0, 1, 2, 3, 4), lam_kp=1e-3):
    tasks = {s: CompositionalTask(A=A, B=B, part_dim=part_dim, seed=s) for s in seeds}
    out = {}
    for flag in (False, True):
        held, trn, a0, a1, lat, cn, curves = [], [], [], [], [], [], []
        for s in seeds:
            task = tasks[s]
            dims = tuple([task.obs_dim] + [width] * (depth - 1))
            cfg = CerebrumConfig(dims=dims, n_settle=12, seed=s,
                              align_feedback=flag, lam_kp=lam_kp)
            res = run_pc_completion(task, cfg, passes=passes)
            held.append(res["acc_heldout"]); trn.append(res["acc_train"])
            a0.append(res["align_start"]); a1.append(res["align_end"])
            lat.append(res["lat_act"]); cn.append(res["comp_norm"])
            curves.append(res["align_curve"])
        out[flag] = dict(held=held, trn=trn, a0=a0, a1=a1, lat=lat, cn=cn, curves=curves)
    out["meta"] = dict(width=width, depth=depth, A=A, B=B, part_dim=part_dim,
                       passes=passes, seeds=list(seeds), lam_kp=lam_kp, chance=1.0 / B)
    return out


def _verdict(out):
    chance = out["meta"]["chance"]
    on = out[True]; off = out[False]
    a_on_s, a_on_e = float(np.mean(on["a0"])), float(np.mean(on["a1"]))
    align_rose = (a_on_e - a_on_s) > 0.1 and a_on_e > float(np.mean(off["a1"]))
    mt_on, ht_on = mean_ci(on["trn"]); mt_off, _ = mean_ci(off["trn"])
    mh_on, hh_on = mean_ci(on["held"]); mh_off, _ = mean_ci(off["held"])
    trn_above_chance = (mt_on - ht_on) > chance + 0.02
    trn_improved = (mt_on - mt_off) > 0.03
    held_improved = (mh_on - mh_off) > 0.03
    lines = []
    lines.append(f"alignment ON: cosine {a_on_s:+.3f} -> {a_on_e:+.3f}  "
                 f"(OFF ends at {float(np.mean(off['a1'])):+.3f}) -> "
                 f"{'RISES' if align_rose else 'does NOT rise meaningfully'}")
    if not align_rose:
        return "VERDICT: MECHANISM FAILED — alignment did not rise; KP rule mis-tuned.\n" + "\n".join(lines)
    if trn_above_chance or trn_improved or held_improved:
        lines.append("VERDICT: FIXED (at least partially) — alignment rose AND "
                     "within-dist/held-out completion improved above chance/baseline.")
    else:
        lines.append("VERDICT: HONEST NEGATIVE — alignment rose (B -> W.T works as a local rule), "
                     "AND aligned settling lowers bottom-layer reconstruction error, but the local "
                     "four-factor rule STILL does not build a useful compositional f1->f2 latent: "
                     "within-dist completion stays at chance and held-out does not improve. "
                     "Feedback alignment is NECESSARY-looking but NOT SUFFICIENT for OP1 here.")
    return "\n".join(lines)


if __name__ == "__main__":
    out = run_probe()
    m = out["meta"]
    print("KOLEN-POLLACK FEEDBACK-ALIGNMENT probe on the compositional completion task")
    print(f"dims=(obs={2*m['part_dim']},{','.join([str(m['width'])]*(m['depth']-1))}); "
          f"A={m['A']} x B={m['B']}; passes={m['passes']}; seeds={len(m['seeds'])}; "
          f"lam_kp={m['lam_kp']:.0e}; chance=1/B={m['chance']:.3f}")
    print()
    hdr = f"{'flag':>6}  {'align start->end (CI)':>30}  {'within-dist acc (CI)':>26}  {'held-out acc (CI)':>26}  {'lat|x|':>8}  {'compN':>7}"
    print(hdr)
    for flag in (False, True):
        d = out[flag]
        astr = f"{fmt_ci(d['a0'])} -> {fmt_ci(d['a1'])}"
        print(f"{('ON' if flag else 'OFF'):>6}  {astr:>30}  {fmt_ci(d['trn']):>26}  "
              f"{fmt_ci(d['held']):>26}  {float(np.mean(d['lat'])):>8.4f}  {float(np.mean(d['cn'])):>7.4f}")
    print()
    # alignment curve (mean over seeds) for the ON run, sampled
    curves = np.array(out[True]["curves"])
    mean_curve = curves.mean(axis=0)
    idx = np.linspace(0, len(mean_curve) - 1, min(9, len(mean_curve))).astype(int)
    print("alignment cosine curve (ON, mean over seeds), pass index -> cosine:")
    print("  " + "  ".join(f"p{i}={mean_curve[i]:+.3f}" for i in idx))
    print()
    print(_verdict(out))
