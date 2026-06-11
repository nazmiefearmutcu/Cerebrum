import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.tasks.continual import run_continual
from benchmarks.baselines.ewc import run_continual_ewc
from benchmarks.stats import mean_ci, fmt_ci

if __name__ == "__main__":
    # I4b-ForgetRobust: >=8 seeds, a SINGLE FIXED knob set (no per-task retuning), and a
    # noise-free (T=0) measurement readout so forgetA reflects the learned weights rather
    # than the stochastic settling floor. With these the fuse vs always-plastic CIs separate.
    seeds = (0, 1, 2, 3, 4, 5, 6, 7)

    def runs(fn, **kw):
        return [fn(seed=s, **kw) for s in seeds]

    fuse = runs(run_continual, use_fuse=True)
    plastic = runs(run_continual, use_fuse=False)
    ewc = runs(run_continual_ewc)
    col = lambda rs, k: [r[k] for r in rs]
    n = len(seeds)
    print(f"Stage-3 catastrophic forgetting (mean +/- 95% CI over {n} seeds; lower forgetA is better)")
    print(f"{'method':<16}{'forgetA':>20}{'errC_afterC':>20}")
    print(f"{'CEREBRUM-fuse':<16}{fmt_ci(col(fuse,'forgetA')):>20}{fmt_ci(col(fuse,'errC_afterC')):>20}"
          f"   (cbar={mean_ci(col(fuse,'cbar'))[0]:.2f})")
    print(f"{'always-plastic':<16}{fmt_ci(col(plastic,'forgetA')):>20}{fmt_ci(col(plastic,'errC_afterC')):>20}")
    print(f"{'EWC-analog':<16}{fmt_ci(col(ewc,'forgetA')):>20}{fmt_ci(col(ewc,'errC_afterC')):>20}"
          f"   (+Fisher pass +anchors)")

    # robustness verdict (single fixed knob set, no per-task retuning)
    mf, hf = mean_ci(col(fuse, 'forgetA'))
    mp, hp = mean_ci(col(plastic, 'forgetA'))
    every_lower = all(f < p for f, p in zip(col(fuse, 'forgetA'), col(plastic, 'forgetA')))
    ci_sep = (mf + hf) < (mp - hp)
    print()
    print(f"robustness ({n} seeds, single fixed knob set, T=0 noise-free eval):")
    print(f"  fuse < always-plastic on every seed : {every_lower} ({sum(f<p for f,p in zip(col(fuse,'forgetA'),col(plastic,'forgetA')))}/{n})")
    print(f"  95% CIs separated (fuse upper {mf+hf:.3f} < plastic lower {mp-hp:.3f}) : {ci_sep}")
