import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.tasks.continual import run_continual
from benchmarks.baselines.ewc import run_continual_ewc
from benchmarks.stats import mean_ci, fmt_ci

if __name__ == "__main__":
    seeds = (0, 1, 2, 3, 4)

    def runs(fn, **kw):
        return [fn(seed=s, **kw) for s in seeds]

    fuse = runs(run_continual, use_fuse=True)
    plastic = runs(run_continual, use_fuse=False)
    ewc = runs(run_continual_ewc)
    col = lambda rs, k: [r[k] for r in rs]
    print("Stage-3 catastrophic forgetting (mean +/- 95% CI over 5 seeds; lower forgetA is better)")
    print(f"{'method':<16}{'forgetA':>20}{'errC_afterC':>20}")
    print(f"{'GRAIL-fuse':<16}{fmt_ci(col(fuse,'forgetA')):>20}{fmt_ci(col(fuse,'errC_afterC')):>20}"
          f"   (cbar={mean_ci(col(fuse,'cbar'))[0]:.2f})")
    print(f"{'always-plastic':<16}{fmt_ci(col(plastic,'forgetA')):>20}{fmt_ci(col(plastic,'errC_afterC')):>20}")
    print(f"{'EWC-analog':<16}{fmt_ci(col(ewc,'forgetA')):>20}{fmt_ci(col(ewc,'errC_afterC')):>20}"
          f"   (+Fisher pass +anchors)")
