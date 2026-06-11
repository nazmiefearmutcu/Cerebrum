import sys
import os
import types
import importlib.util

submission_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "cerebrum_submission.py"))

if os.path.exists(submission_path):
    spec = importlib.util.spec_from_file_location("cerebrum", submission_path)
    cerebrum_mod = importlib.util.module_from_spec(spec)
    sys.modules["cerebrum"] = cerebrum_mod
    spec.loader.exec_module(cerebrum_mod)
    
    submodule_names = [
        "config", "counters", "types", "invariants", "grid_head", "neuromod",
        "nonlinear", "pc_core", "plasticity", "rng", "core_net", "gate",
        "metaplasticity", "workspace", "unified", "workspace_net", "energy", "grounding"
    ]
    
    for name in submodule_names:
        full_name = f"cerebrum.{name}"
        submod = types.ModuleType(full_name)
        for key, val in cerebrum_mod.__dict__.items():
            setattr(submod, key, val)
        sys.modules[full_name] = submod
        setattr(cerebrum_mod, name, submod)

sys.path.insert(0, os.path.dirname(__file__))
