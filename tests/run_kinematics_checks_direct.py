import sys
import os
import traceback
import numpy as np

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from physical_validation import Geometric3DOFSolver, joint_to_ticks, torque_to_current

def test_solver_nan_inputs():
    solver = Geometric3DOFSolver(l1=1.0, l2=1.0, l3=1.0)
    for x, y, z in [(np.nan, 0.0, 1.0), (0.0, np.nan, 1.0), (0.0, 0.0, np.nan), (np.nan, np.nan, np.nan)]:
        angles, reachable = solver.inverse_kinematics(x, y, z)
        assert reachable is False
        assert angles == (0.0, 0.0, 0.0)

def test_solver_inf_inputs():
    solver = Geometric3DOFSolver(l1=1.0, l2=1.0, l3=1.0)
    for x, y, z in [(np.inf, 0.0, 1.0), (0.0, -np.inf, 1.0), (0.0, 0.0, np.inf), (-np.inf, np.inf, -np.inf)]:
        angles, reachable = solver.inverse_kinematics(x, y, z)
        assert reachable is False, f"Infinite target ({x}, {y}, {z}) should not be reachable"

def test_solver_extreme_values():
    solver = Geometric3DOFSolver(l1=1.0, l2=1.0, l3=1.0)
    angles, reachable = solver.inverse_kinematics(1e300, 1e300, 1e300)
    assert not reachable
    assert angles == (0.0, 0.0, 0.0)

def test_solver_singularity_signed_zeros():
    solver = Geometric3DOFSolver(l1=1.0, l2=1.0, l3=1.0)
    for x, y in [(0.0, 0.0), (-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0)]:
        angles, reachable = solver.inverse_kinematics(x, y, 2.0)
        assert angles[0] == 0.0, f"Base angle should lock to 0.0 for signed zeros: x={x}, y={y}"

def test_solver_precision_boundaries():
    solver = Geometric3DOFSolver(l1=1.0, l2=1.0, l3=1.0)
    max_reach = solver.l2 + solver.l3
    
    angles_out, reachable_out = solver.inverse_kinematics(max_reach + 1e-5, 0.0, 1.0)
    assert not reachable_out
    
    angles_in, reachable_in = solver.inverse_kinematics(max_reach - 1e-5, 0.0, 1.0)
    assert reachable_in

def test_joint_to_ticks_nan_inf():
    assert joint_to_ticks(np.nan) == 0
    assert joint_to_ticks(np.inf) == 0

def test_torque_to_current_nan_inf():
    assert np.isnan(torque_to_current(np.nan))
    assert torque_to_current(np.inf, max_current=10.0) == 10.0
    assert torque_to_current(-np.inf, max_current=10.0) == -10.0

def test_torque_to_current_negative_max_current():
    res = torque_to_current(0.0, torque_constant=1.0, max_current=-5.0)
    assert res == 0.0, f"Expected 0.0 under corrected behavior, got {res}"

def test_torque_to_current_zero_division():
    assert torque_to_current(10.0, torque_constant=1e-10) == 0.0
    assert torque_to_current(10.0, torque_constant=-1e-10) == 0.0
    assert torque_to_current(10.0, torque_constant=0.0) == 0.0

def test_torque_to_current_large_kt():
    res = torque_to_current(10.0, torque_constant=1e10, max_current=10.0)
    assert abs(res - 1e-9) < 1e-12

def main():
    tests = [
        test_solver_nan_inputs,
        test_solver_inf_inputs,
        test_solver_extreme_values,
        test_solver_singularity_signed_zeros,
        test_solver_precision_boundaries,
        test_joint_to_ticks_nan_inf,
        test_torque_to_current_nan_inf,
        test_torque_to_current_negative_max_current,
        test_torque_to_current_zero_division,
        test_torque_to_current_large_kt
    ]
    
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}")
            traceback.print_exc()
            failed += 1
            
    print(f"Total: {passed} passed, {failed} failed")

if __name__ == "__main__":
    main()
