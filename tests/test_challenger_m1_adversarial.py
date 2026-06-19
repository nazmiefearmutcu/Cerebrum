import pytest
import numpy as np
from physical_validation import Geometric3DOFSolver, joint_to_ticks, torque_to_current

# ==========================================
# 1. Geometric3DOFSolver Adversarial Tests
# ==========================================

def test_solver_nan_coordinates():
    solver = Geometric3DOFSolver(l1=1.0, l2=1.0, l3=1.0)
    
    # Test cases with NaN in coordinates
    nan_cases = [
        (np.nan, 1.0, 1.0),
        (1.0, np.nan, 1.0),
        (1.0, 1.0, np.nan),
        (np.nan, np.nan, np.nan)
    ]
    for x, y, z in nan_cases:
        angles, reachable = solver.inverse_kinematics(x, y, z)
        print(f"NaN Target ({x}, {y}, {z}) -> reachable: {reachable}, angles: {angles}")
        assert reachable is False, "NaN coordinates should be unreachable"
        assert angles == (0.0, 0.0, 0.0), "NaN coordinates should return (0.0, 0.0, 0.0) angles"

def test_solver_inf_coordinates():
    solver = Geometric3DOFSolver(l1=1.0, l2=1.0, l3=1.0)
    
    # Test cases with infinity in coordinates
    inf_cases = [
        (np.inf, 1.0, 1.0),
        (1.0, -np.inf, 1.0),
        (1.0, 1.0, np.inf),
        (-np.inf, np.inf, -np.inf)
    ]
    for x, y, z in inf_cases:
        angles, reachable = solver.inverse_kinematics(x, y, z)
        assert reachable is False, f"Infinite target ({x}, {y}, {z}) must be unreachable"
        assert angles == (0.0, 0.0, 0.0), "Infinite coordinates should return (0.0, 0.0, 0.0) angles"

def test_solver_overflow_extremes():
    solver = Geometric3DOFSolver(l1=1.0, l2=1.0, l3=1.0)
    
    angles, reachable = solver.inverse_kinematics(1e160, 1e160, 1e160)
    assert reachable is False
    assert angles == (0.0, 0.0, 0.0)


def test_solver_signed_zeros():
    solver = Geometric3DOFSolver(l1=1.0, l2=1.0, l3=1.0)
    
    # Singularity handling should work for all signed zero combinations
    for x, y in [(0.0, 0.0), (-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0)]:
        angles, reachable = solver.inverse_kinematics(x, y, 2.0)
        assert angles[0] == 0.0, f"Base angle must lock to 0.0 for signed zeros: x={x}, y={y}"

def test_solver_precision_boundaries():
    solver = Geometric3DOFSolver(l1=1.0, l2=1.0, l3=1.0)
    max_reach = solver.l2 + solver.l3 # 2.0
    
    # 1. Target slightly outside max reach
    _, reachable_out = solver.inverse_kinematics(max_reach + 1e-9, 0.0, 1.0)
    assert reachable_out is False, "Should be unreachable outside max reach"
    
    # 2. Target slightly inside max reach
    _, reachable_in = solver.inverse_kinematics(max_reach - 1e-9, 0.0, 1.0)
    assert reachable_in is True, "Should be reachable inside max reach"

def test_solver_link_lengths_boundary():
    # What if link lengths are zero?
    solver_zero = Geometric3DOFSolver(l1=0.0, l2=0.0, l3=0.0)
    angles, reachable = solver_zero.inverse_kinematics(1.0, 1.0, 1.0)
    assert reachable is False
    # Check that it doesn't crash with zero-division
    assert np.isnan(angles).any() # division by zero in cos_val numerator/denominator results in NaN

# ==========================================
# 2. joint_to_ticks Adversarial Tests
# ==========================================

def test_joint_to_ticks_extremes():
    # Large inputs
    assert joint_to_ticks(1e9) == 1000000000000
    assert joint_to_ticks(-1e9) == -1000000000000
    
    # Sub-tick inputs
    assert joint_to_ticks(1e-6) == 0
    assert joint_to_ticks(-1e-6) == 0

def test_joint_to_ticks_rounding_boundaries():
    # Test standard round half-to-even behavior or standard round behavior
    # Python 3 round() rounds to nearest even number for half cases:
    # e.g., round(0.5) = 0, round(1.5) = 2.
    # Let's verify what joint_to_ticks returns.
    # ticks = joint_angle * 1000.0
    # For angle = 0.0005: ticks = 0.5. round(0.5) = 0.
    assert joint_to_ticks(0.0005) == 0
    # For angle = 0.0015: ticks = 1.5. round(1.5) = 2.
    assert joint_to_ticks(0.0015) == 2
    # For angle = 0.0025: ticks = 2.5. round(2.5) = 2.
    assert joint_to_ticks(0.0025) == 2
    # For angle = 0.0035: ticks = 3.5. round(3.5) = 4.
    assert joint_to_ticks(0.0035) == 4

def test_joint_to_ticks_nan_inf_exceptions():
    assert joint_to_ticks(np.nan) == 0
    assert joint_to_ticks(np.inf) == 0

# ==========================================
# 3. torque_to_current Adversarial Tests
# ==========================================

def test_torque_to_current_zero_kt():
    # zero-division protection: Kt < 1e-9 should return 0.0
    assert torque_to_current(10.0, torque_constant=0.0) == 0.0
    assert torque_to_current(10.0, torque_constant=1e-10) == 0.0
    assert torque_to_current(10.0, torque_constant=-1e-10) == 0.0

def test_torque_to_current_negative_max_current_bug():
    res = torque_to_current(0.0, torque_constant=1.0, max_current=-5.0)
    assert res == 0.0, f"Expected 0.0 current for negative max current bounds, got {res}"

def test_torque_to_current_limits():
    # Hard limit of 25.0A must be respected even if max_current is larger
    assert torque_to_current(100.0, torque_constant=1.0, max_current=50.0) == 25.0
    assert torque_to_current(-100.0, torque_constant=1.0, max_current=50.0) == -25.0
    
    # Soft clamp max_current is respected if it is smaller than 25.0
    assert torque_to_current(100.0, torque_constant=1.0, max_current=10.0) == 10.0
    assert torque_to_current(-100.0, torque_constant=1.0, max_current=10.0) == -10.0

def test_torque_to_current_nan_inf():
    # NaN in torque -> returns NaN
    assert np.isnan(torque_to_current(np.nan, torque_constant=1.0))
    
    # Inf in torque -> correctly clamps to effective limit
    assert torque_to_current(np.inf, torque_constant=1.0, max_current=10.0) == 10.0
    assert torque_to_current(-np.inf, torque_constant=1.0, max_current=10.0) == -10.0
