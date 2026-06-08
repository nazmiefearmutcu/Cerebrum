"""Tiny stats helpers for benchmark reporting — pure NumPy, no scipy.

Reports mean +/- a two-sided 95% confidence interval over seeds, using a small-sample
Student-t critical value (so n=5 is not over-confident). For df>30 we fall back to the
normal approximation (1.96)."""
import numpy as np

# two-sided 95% Student-t critical values by degrees of freedom (n-1)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}


def _t95(df):
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    if df > 30:
        return 1.96
    # nearest tabulated df at or above (conservative)
    for k in sorted(_T95):
        if k >= df:
            return _T95[k]
    return 1.96


def mean_ci(values):
    """Return (mean, half_width_95). half_width is the +/- CI margin; 0.0 for n<2."""
    a = np.asarray(list(values), dtype=float)
    n = a.size
    mean = float(np.mean(a)) if n else float("nan")
    if n < 2:
        return mean, 0.0
    sd = float(np.std(a, ddof=1))          # sample standard deviation
    se = sd / np.sqrt(n)                    # standard error of the mean
    return mean, float(_t95(n - 1) * se)


def fmt_ci(values, prec=3):
    """'mean +/- ci' string."""
    m, h = mean_ci(values)
    return f"{m:.{prec}f} +/- {h:.{prec}f}"
