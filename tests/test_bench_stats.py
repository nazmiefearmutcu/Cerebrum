import numpy as np
from benchmarks.stats import mean_ci, fmt_ci


def test_mean_ci_basic():
    m, h = mean_ci([1.0, 1.0, 1.0, 1.0, 1.0])
    assert abs(m - 1.0) < 1e-12 and h == 0.0          # zero variance -> zero width

def test_mean_ci_single_value_zero_width():
    m, h = mean_ci([0.7])
    assert m == 0.7 and h == 0.0                       # n<2 -> no CI

def test_mean_ci_positive_width_with_spread():
    m, h = mean_ci([0.2, 0.4, 0.6, 0.8, 1.0])
    assert abs(m - 0.6) < 1e-9 and h > 0.0             # spread -> positive CI half-width

def test_mean_ci_matches_hand_computation_n5():
    vals = [0.2, 0.4, 0.6, 0.8, 1.0]                   # sd(ddof=1)=0.31623, se=0.14142, t4=2.776
    m, h = mean_ci(vals)
    assert abs(h - 2.776 * (np.std(vals, ddof=1)/np.sqrt(5))) < 1e-9

def test_fmt_ci_string():
    s = fmt_ci([0.5, 0.5, 0.5])
    assert s.startswith("0.500") and "+/-" in s
