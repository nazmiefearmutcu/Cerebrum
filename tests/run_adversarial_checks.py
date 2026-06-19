import sys
import os
import traceback
import numpy as np

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tests.test_challenger_kinematics as tck

def run_tests():
    test_functions = [
        tck.test_solver_nan_inputs,
        tck.test_solver_inf_inputs,
        tck.test_solver_extreme_values,
        tck.test_solver_singularity_signed_zeros,
        tck.test_solver_precision_boundaries,
        tck.test_joint_to_ticks_nan_inf,
        tck.test_torque_to_current_nan_inf,
        tck.test_torque_to_current_negative_max_current,
        tck.test_torque_to_current_zero_division,
        tck.test_torque_to_current_large_kt
    ]
    
    passed_count = 0
    failed_count = 0
    results = []
    
    print("Running adversarial tests...")
    print("=" * 60)
    for func in test_functions:
        name = func.__name__
        try:
            func()
            print(f"[PASS] {name}")
            results.append((name, "PASS", None))
            passed_count += 1
        except Exception as e:
            err_msg = traceback.format_exc()
            print(f"[FAIL] {name}")
            print(err_msg)
            results.append((name, "FAIL", err_msg))
            failed_count += 1
            
    print("=" * 60)
    print(f"Summary: {passed_count} passed, {failed_count} failed")
    
    # Exit with code 1 if any tests failed, so we know it failed
    if failed_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    run_tests()
