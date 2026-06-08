import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.tasks.binding import run_binding
from benchmarks.baselines.soft_mixer import run_binding_soft
from benchmarks.stats import fmt_ci

if __name__ == "__main__":
    seeds = (0, 1, 2, 3, 4)
    print("== Stage-2A: ROUTING (improved gate: low temp + Go-decay; mean +/- 95% CI over 5 seeds) ==")
    for nm in (4, 6):
        hard = [run_binding(n_modules=nm, k_slots=1, trials=500, seed=s) for s in seeds]
        print(f"[M={nm}] (chance={1.0/nm:.3f})  one-hot routing_acc = {fmt_ci([h['routing_acc'] for h in hard])}"
              f"  | win_entropy = {fmt_ci([h['win_entropy'] for h in hard])}")
    print("\n== Stage-2B: WRITE-RULE ablation at MODERATE gate temp (gate_temp=0.5, where P spreads) ==")
    print("   (proves one-hot is load-bearing: soft write blends >1 module per slot = gated-SSM collapse)")
    for nm in (4, 6):
        hard = [run_binding(n_modules=nm, k_slots=1, trials=500, seed=s, gate_temp=0.5) for s in seeds]
        soft = [run_binding_soft(n_modules=nm, k_slots=1, trials=500, seed=s, gate_temp=0.5) for s in seeds]
        print(f"[M={nm}] one-hot routing = {fmt_ci([h['routing_acc'] for h in hard])} (participation=1.0)"
              f"  | soft routing = {fmt_ci([s['routing_acc'] for s in soft])}"
              f"  participation = {fmt_ci([s['mean_slot_participation'] for s in soft], prec=2)}")
