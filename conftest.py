"""FOOTGUN, READ THIS FIRST.

This file does NOT just add the repo root to `sys.path`. If `cerebrum_submission.py`
exists, it loads that single generated file and registers it in `sys.modules` under the
name `cerebrum`, plus a fake submodule for every `cerebrum.*` name below (each one is a
shallow alias whose namespace is the bundle's namespace).

Consequence: for the entire test run, `import cerebrum...` resolves to the ~122 KB
generated bundle, NOT to the `cerebrum/` package directory. Editing a file under
`cerebrum/` therefore has NO effect on the test suite until you re-run
`python3 build_submission.py`.

That is only sound while the bundle is exactly what `build_submission.py` emits from the
current package. `tests/test_submission_bundle_is_current.py` rebuilds the bundle in
memory and fails loudly if the committed file has drifted, so a stale bundle can no
longer pass silently. Do not delete that test without also deleting this rebinding.
"""

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
        "metaplasticity", "workspace", "unified", "workspace_net", "energy", "grounding",
        "hippocampus", "grounding.vlm_adapter"
    ]
    
    for name in submodule_names:
        full_name = f"cerebrum.{name}"
        submod = types.ModuleType(full_name)
        for key, val in cerebrum_mod.__dict__.items():
            setattr(submod, key, val)
        sys.modules[full_name] = submod
        
        # Link to parent module
        parts = name.split(".")
        curr = cerebrum_mod
        for p in parts[:-1]:
            if not hasattr(curr, p):
                setattr(curr, p, types.ModuleType(f"cerebrum.{p}"))
            curr = getattr(curr, p)
        setattr(curr, parts[-1], submod)

sys.path.insert(0, os.path.dirname(__file__))
