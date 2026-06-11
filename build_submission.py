import re

files_order = [
    "cerebrum/config.py",
    "cerebrum/counters.py",
    "cerebrum/types.py",
    "cerebrum/invariants.py",
    "cerebrum/nonlinear.py",
    "cerebrum/rng.py",
    "cerebrum/pc_core.py",
    "cerebrum/grid_head.py",
    "cerebrum/gate.py",
    "cerebrum/workspace.py",
    "cerebrum/neuromod.py",
    "cerebrum/metaplasticity.py",
    "cerebrum/plasticity.py",
    "cerebrum/hippocampus.py",
    "cerebrum/unified.py",
    "cerebrum/workspace_net.py",
    "cerebrum/core_net.py",
    "cerebrum/energy.py",
    "cerebrum/grounding/sensory.py",
    "cerebrum/grounding/vlm_adapter.py",
    "cerebrum/grounding/motor.py",
    "cerebrum/grounding/physics.py",
    "cerebrum/grounding/ros_node.py",
    "cerebrum/grounding/reflex.py",
    "cerebrum/grounding/__init__.py"
]

header = """__version__ = "0.0.1"

import numpy as np
import torch
from dataclasses import dataclass, field, replace
"""

content = [header]

for fp in files_order:
    with open(fp, "r") as f:
        lines = f.readlines()
    filtered_lines = []
    for line in lines:
        # Filter out local imports
        if re.search(r"from\s+\.+(config|counters|types|invariants|nonlinear|rng|pc_core|grid_head|gate|workspace|neuromod|metaplasticity|plasticity|unified|workspace_net|core_net|energy|invariants|sensory|motor|physics|ros_node|reflex|grounding|hippocampus|vlm_adapter)", line):
            continue
        if re.match(r"^(import\s+numpy|import\s+torch|from\s+dataclasses|import\s+types|from\s+datetime)", line):
            continue
        filtered_lines.append(line)
    content.append(f"\n# ==========================================\n# {fp}\n# ==========================================\n")
    content.append("".join(filtered_lines))

# Replace escape sequences
merged = "\\n".join(content)
merged = merged.replace("\\n", "\n")

with open("cerebrum_submission.py", "w") as f:
    f.write(merged)

print("Merged successfully!")
