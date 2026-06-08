import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from benchmarks.tasks.continual import run_continual
from benchmarks.baselines.ewc import run_continual_ewc

if __name__ == "__main__":
    seeds = (0, 1, 2)

    def avg(fn, **kw):
        rs = [fn(seed=s, **kw) for s in seeds]
        return {k: float(np.mean([r[k] for r in rs])) for k in rs[0] if isinstance(rs[0][k], (int, float))}

    fuse = avg(run_continual, use_fuse=True)
    plastic = avg(run_continual, use_fuse=False)
    ewc = avg(run_continual_ewc)
    print(f"{'method':<16}{'forgetA':>10}{'errC_afterC':>14}")
    print(f"{'GRAIL-fuse':<16}{fuse['forgetA']:>10.3f}{fuse['errC_afterC']:>14.3f}   (cbar={fuse['cbar']:.2f})")
    print(f"{'always-plastic':<16}{plastic['forgetA']:>10.3f}{plastic['errC_afterC']:>14.3f}")
    print(f"{'EWC-analog':<16}{ewc['forgetA']:>10.3f}{ewc['errC_afterC']:>14.3f}   (+Fisher pass +anchors)")
