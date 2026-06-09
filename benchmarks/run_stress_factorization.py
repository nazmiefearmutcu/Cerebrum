import os
os.environ["GRAIL_BALANCE_GRID_PRECISION"] = "1"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.run_factorization_pipeline import run_sweep, _print_block, _verdict

def run_stress_test():
    configurations = [
        {"name": "Default Shape (A=6, B=6)", "A": 6, "B": 6, "part_dim": 8},
        {"name": "Larger Equal Shape (A=8, B=8)", "A": 8, "B": 8, "part_dim": 8},
        {"name": "Asymmetric Shape (A=4, B=8)", "A": 4, "B": 8, "part_dim": 8},
        {"name": "Asymmetric Shape (A=8, B=4)", "A": 8, "B": 4, "part_dim": 8},
        {"name": "Much Larger Equal Shape (A=10, B=10)", "A": 10, "B": 10, "part_dim": 8},
        {"name": "Wider Embeddings (A=6, B=6, part_dim=12)", "A": 6, "B": 6, "part_dim": 12},
    ]

    for config in configurations:
        print("=" * 80)
        print(f"STRESS TEST: {config['name']}")
        print("=" * 80)
        out = run_sweep(
            A=config["A"],
            B=config["B"],
            part_dim=config["part_dim"],
            width=24,
            depth=3,
            frac_heldout=0.3,
            passes=60,
            seeds=(0, 1, 2)  # Use 3 seeds to run faster but still get CIs
        )
        _print_block(out)
        print(_verdict(out))
        print()

if __name__ == "__main__":
    run_stress_test()
