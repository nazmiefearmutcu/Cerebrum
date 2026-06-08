import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.tasks.binding import run_binding
from benchmarks.baselines.soft_mixer import run_binding_soft
from benchmarks.stats import fmt_ci

if __name__ == "__main__":
    seeds = (0, 1, 2, 3, 4)
    print("Stage-2 emergent routing (mean +/- 95% CI over 5 seeds)")
    for nm in (4, 6):
        hard = [run_binding(n_modules=nm, k_slots=1, trials=500, seed=s) for s in seeds]
        soft = [run_binding_soft(n_modules=nm, k_slots=1, trials=500, seed=s) for s in seeds]
        print(f"[M={nm}] (chance={1.0/nm:.3f})")
        print(f"   one-hot routing_acc = {fmt_ci([h['routing_acc'] for h in hard])}"
              f"  | win_entropy = {fmt_ci([h['win_entropy'] for h in hard])}")
        print(f"   soft    routing_acc = {fmt_ci([s['routing_acc'] for s in soft])}"
              f"  | slot_participation = {fmt_ci([s['mean_slot_participation'] for s in soft], prec=2)}")
