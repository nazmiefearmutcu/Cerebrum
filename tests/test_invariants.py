import numpy as np, pytest
from grail.invariants import assert_one_hot, assert_scalar_M, assert_exogenous_action
from grail.types import Exogenous

def test_one_hot_passes_for_one_hot_columns():
    z = np.array([[1.0,0.0],[0.0,1.0],[0.0,0.0]])  # per-slot (column) one-hot; a slot may be empty
    assert_one_hot(z, axis=0)  # no raise

def test_one_hot_rejects_soft_weights():
    z = np.array([[0.6,0.1],[0.4,0.9]])            # soft mixing weights = BAN-1 violation
    with pytest.raises(AssertionError):
        assert_one_hot(z, axis=0)

def test_scalar_M_passes_and_vector_raises():
    assert_scalar_M(0.7); assert_scalar_M(np.float64(0.7))
    with pytest.raises(AssertionError):
        assert_scalar_M(np.array([0.1, 0.2]))       # vector global signal = DFA = BAN-2

def test_exogenous_action_accepts_only_wrapped():
    assert_exogenous_action(Exogenous(np.array([1.0,0.0])))  # ok
    with pytest.raises(TypeError):
        assert_exogenous_action(np.array([1.0,0.0]))         # plain (possibly data-derived) array rejected
