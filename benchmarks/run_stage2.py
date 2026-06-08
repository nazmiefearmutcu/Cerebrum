import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.tasks.binding import run_binding
from benchmarks.baselines.soft_mixer import run_binding_soft

if __name__ == "__main__":
    for nm in (4, 6):
        hard = run_binding(n_modules=nm, k_slots=1, trials=500, seed=0)
        soft = run_binding_soft(n_modules=nm, k_slots=1, trials=500, seed=0)
        print(f"[M={nm}] one-hot: routing_acc={hard['routing_acc']:.3f} entropy={hard['win_entropy']:.3f} "
              f"(chance={1.0/nm:.3f}) | soft: routing_acc={soft['routing_acc']:.3f} "
              f"participation={soft['mean_slot_participation']:.2f}")
