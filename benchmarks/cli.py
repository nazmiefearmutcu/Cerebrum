#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

# Mapping of task names to script files in the benchmarks directory
TASKS = {
    "task1": "run_task1.py",
    "stage2": "run_stage2.py",
    "routing": "run_stage2.py",
    "stage3": "run_stage3.py",
    "continual": "run_stage3.py",
    "scaling": "run_scaling.py",
    "energy": "run_energy.py",
    "compositional": "run_compositional.py",
    "factorization": "run_factorization.py",
    "factorization_pipeline": "run_factorization_pipeline.py",
    "factorization_multi": "run_factorization_multi.py",
    "factorization_splits": "run_factorization_splits.py",
    "largegraph": "run_largegraph.py",
    "relational": "run_relational.py",
    "transitive": "run_transitive.py",
    "uncertainty": "run_uncertainty.py",
    "align_feedback": "run_align_feedback.py",
    "pillar4_ablation": "run_pillar4_ablation.py",
    "stress_factorization": "run_stress_factorization.py"
}

def main():
    parser = argparse.ArgumentParser(
        description="Cerebrum Experiment Benchmark Runner CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--task",
        choices=sorted(list(TASKS.keys())),
        required=True,
        help="The benchmark task/experiment to execute."
    )
    parser.add_argument(
        "--balance-grid",
        action="store_true",
        help="Enable CEREBRUM_BALANCE_GRID_PRECISION environment variable."
    )
    
    args, unknown = parser.parse_known_args()
    
    script_name = TASKS[args.task]
    benchmarks_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(benchmarks_dir, script_name)
    
    if not os.path.exists(script_path):
        print(f"Error: Script {script_name} not found at {script_path}", file=sys.stderr)
        sys.exit(1)
        
    env = os.environ.copy()
    if args.balance_grid:
        env["CEREBRUM_BALANCE_GRID_PRECISION"] = "1"
        
    print(f"--> Running benchmark task: {args.task} ({script_name})...")
    cmd = [sys.executable, script_path] + unknown
    
    try:
        result = subprocess.run(cmd, env=env, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"Error: Task {args.task} failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nExperiment execution interrupted by user.", file=sys.stderr)
        sys.exit(130)

if __name__ == "__main__":
    main()
