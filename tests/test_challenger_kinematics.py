import pytest
import numpy as np
from physical_validation import Geometric3DOFSolver, joint_to_ticks, torque_to_current

def test_solver_nan_inputs():
    solver = Geometric3DOFSolver(l1=1.0, l2=1.0, l3=1.0)
    
    # Target containing NaN coordinates
    for x, y, z in [(np.nan, 0.0, 1.0), (0.0, np.nan, 1.0), (0.0, 0.0, np.nan), (np.nan, np.nan, np.nan)]:
        angles, reachable = solver.inverse_kinematics(x, y, z)
        # BUG: Currently, if D is NaN, both D > max_reach and D < min_reach evaluate to False,
        # so reachable remains True. We assert that NaN target must return reachable=False.
        # This is expected to FAIL with the current implementation.
        assert not reachable, f"NaN target ({x}, {y}, {z}) must return reachable=False"
        assert not np.isnan(angles).any(), f"NaN target should not produce NaN angles if clipped properly (or should be handled)"

def test_solver_inf_inputs():
    solver = Geometric3DOFSolver(l1=1.0, l2=1.0, l3=1.0)
    
    # Target containing infinite coordinates
    for x, y, z in [(np.inf, 0.0, 1.0), (0.0, -np.inf, 1.0), (0.0, 0.0, np.inf), (-np.inf, np.inf, -np.inf)]:
        angles, reachable = solver.inverse_kinematics(x, y, z)
        # infinite coordinates are out of reach, so reachable must be False
        assert not reachable, f"Infinite target ({x}, {y}, {z}) should not be reachable"

def test_solver_extreme_values():
    solver = Geometric3DOFSolver(l1=1.0, l2=1.0, l3=1.0)
    
    angles, reachable = solver.inverse_kinematics(1e300, 1e300, 1e300)
    assert not reachable
    assert angles == (0.0, 0.0, 0.0)

def test_solver_singularity_signed_zeros():
    solver = Geometric3DOFSolver(l1=1.0, l2=1.0, l3=1.0)
    
    # Verify base angle is locked to 0.0 for all signed zero combinations
    for x, y in [(0.0, 0.0), (-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0)]:
        angles, reachable = solver.inverse_kinematics(x, y, 2.0)
        assert angles[0] == 0.0, f"Base angle should lock to 0.0 for signed zeros: x={x}, y={y}"

def test_solver_precision_boundaries():
    solver = Geometric3DOFSolver(l1=1.0, l2=1.0, l3=1.0)
    max_reach = solver.l2 + solver.l3
    
    # Target slightly outside max reach (D > 2.0)
    angles_out, reachable_out = solver.inverse_kinematics(max_reach + 1e-5, 0.0, 1.0)
    assert not reachable_out
    
    # Target slightly inside max reach (D <= 2.0)
    angles_in, reachable_in = solver.inverse_kinematics(max_reach - 1e-5, 0.0, 1.0)
    assert reachable_in

def test_joint_to_ticks_nan_inf():
    assert joint_to_ticks(np.nan) == 0
    assert joint_to_ticks(np.inf) == 0

def test_torque_to_current_nan_inf():
    # If torque is NaN, current becomes NaN. np.clip(nan) is nan.
    assert np.isnan(torque_to_current(np.nan))
    
    # If torque is Inf, it should clamp to the limit
    assert torque_to_current(np.inf, max_current=10.0) == 10.0
    assert torque_to_current(-np.inf, max_current=10.0) == -10.0

def test_torque_to_current_negative_max_current():
    # BUG: If max_current is negative, effective_limit is negative,
    # causing np.clip to receive inverted bounds: np.clip(current, 5.0, -5.0).
    # This leads to incorrect/unexpected outputs.
    # For example, torque_to_current(0.0, max_current=-5.0) will return -5.0 instead of 0.0.
    res = torque_to_current(0.0, torque_constant=1.0, max_current=-5.0)
    assert res == 0.0, f"Expected 0.0 current for 0.0 torque, but got {res}"

def test_torque_to_current_zero_division():
    # Extremely small torque constant (absolute value < 1e-9) should return 0.0
    assert torque_to_current(10.0, torque_constant=1e-10) == 0.0
    assert torque_to_current(10.0, torque_constant=-1e-10) == 0.0
    assert torque_to_current(10.0, torque_constant=0.0) == 0.0

def test_torque_to_current_large_kt():
    # Extremely large Kt should produce very small current but not crash
    res = torque_to_current(10.0, torque_constant=1e10, max_current=10.0)
    assert pytest.approx(res, abs=1e-12) == 1e-9
