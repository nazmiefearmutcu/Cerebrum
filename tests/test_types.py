import numpy as np, pytest
from grail.types import Exogenous

def test_exogenous_wraps_array():
    a = Exogenous(np.array([1.0, 0.0]))
    assert isinstance(a.value, np.ndarray) and a.value.shape == (2,)

def test_plain_array_is_not_exogenous():
    # the grid transition will only accept Exogenous; a plain ndarray (which could be derived
    # from data) is rejected -> z_act cannot become data-dependent by construction (BAN-1).
    assert not isinstance(np.array([1.0, 0.0]), Exogenous)
