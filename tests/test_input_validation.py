import pytest
import numpy as np
import torch
from cerebrum.config import CerebrumConfig
from cerebrum.unified import CerebrumNet
from cerebrum.types import Exogenous

def test_obs_slices_not_list_or_tuple():
    cfg = CerebrumConfig(dims=(4, 8), n_settle=2, seed=42)
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg)
    action = Exogenous(np.array([0.1, 0.1]))
    
    # dict passed as obs_slices
    with pytest.raises(TypeError, match="obs_slices must be a list or tuple of slices."):
        net.step({0: [1.0, 1.0, 1.0, 1.0], 1: [1.0, 1.0, 1.0, 1.0]}, action, reward=1.0)

    # string passed as obs_slices
    with pytest.raises(TypeError, match="obs_slices must be a list or tuple of slices."):
        net.step("invalid", action, reward=1.0)

def test_obs_slices_length_mismatch():
    cfg = CerebrumConfig(dims=(4, 8), n_settle=2, seed=42)
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg)
    action = Exogenous(np.array([0.1, 0.1]))
    
    # Fewer slices (1) than modules (2)
    with pytest.raises(ValueError, match="Number of observation slices.*must match n_modules"):
        net.step([[1.0, 1.0, 1.0, 1.0]], action, reward=1.0)
        
    # More slices (3) than modules (2)
    with pytest.raises(ValueError, match="Number of observation slices.*must match n_modules"):
        net.step([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], action, reward=1.0)

def test_obs_slices_not_1d():
    cfg = CerebrumConfig(dims=(4, 8), n_settle=2, seed=42)
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg)
    action = Exogenous(np.array([0.1, 0.1]))
    
    # 2D slice that has first dimension matching slice_dim (4) but is 4x4
    obs_slices_2d = [
        [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        [1.0, 1.0, 1.0, 1.0]
    ]
    with pytest.raises(ValueError, match="Each observation slice must be a 1D tensor/array."):
        net.step(obs_slices_2d, action, reward=1.0)

    # Flat 1D list passed directly (e.g. elements of obs_slices are floats)
    # This evaluates to iterating over float elements, each being 0-d tensor
    obs_slices_flat_direct = [1.0, 2.0]  # length is 2 (matches M_), but elements are floats (0-d)
    with pytest.raises(ValueError, match="Each observation slice must be a 1D tensor/array."):
        net.step(obs_slices_flat_direct, action, reward=1.0)

def test_obs_slices_correct():
    cfg = CerebrumConfig(dims=(4, 8), n_settle=2, seed=42)
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg)
    action = Exogenous(np.array([0.1, 0.1]))
    
    valid_obs_slices = [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]
    # Should not raise any error
    z, M = net.step(valid_obs_slices, action, reward=1.0)
    assert M is not None
    assert z is not None
