import pytest
import numpy as np
import torch
from cerebrum.config import CerebrumConfig
from cerebrum.unified import CerebrumNet
from cerebrum.types import Exogenous

def test_nan_leakage_list_obs():
    cfg = CerebrumConfig(dims=(4, 8), n_settle=2, seed=42)
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg)
    
    # Verify weights are finite initially
    for m in net.modules:
        for w in m.W:
            assert torch.isfinite(w).all()
            
    # List containing nan
    obs_slices_nan = [[1.0, float('nan'), 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    action = Exogenous(np.array([0.1, 0.1]))
    
    # Try stepping
    try:
        net.step(obs_slices_nan, action, reward=1.0)
    except Exception as e:
        pytest.fail(f"step crashed with exception: {e}")
        
    # Check if there are NaNs in weights
    for m in net.modules:
        for w in m.W:
            assert torch.isfinite(w).all(), "NaN/Inf leaked into weights via list observations!"
