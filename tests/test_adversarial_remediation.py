import pytest
import numpy as np
import torch
from cerebrum.config import CerebrumConfig
from cerebrum.unified import CerebrumNet
from cerebrum.types import Exogenous

def test_vulnerability_1_multidimensional_observations():
    """
    Vulnerability 1: Passing nested multi-dimensional arrays as observation slices
    which could bypass the length check because len(2D_array) == first dimension,
    but later crash during PyTorch operations.
    """
    cfg = CerebrumConfig(dims=(4, 8), n_settle=2, seed=42)
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg)
    action = Exogenous(np.array([0.1, 0.1]))
    
    # 2D observation slice (shape 4x4) where len() is 4, matching slice_dim,
    # but ndim is 2.
    obs_slices_2d = [
        np.ones((4, 4)),
        np.ones(4)
    ]
    with pytest.raises(ValueError, match="Each observation slice must be a 1D tensor/array."):
        net.step(obs_slices_2d, action, reward=1.0)

    # 3D observation slice
    obs_slices_3d = [
        np.ones((4, 4, 4)),
        np.ones(4)
    ]
    with pytest.raises(ValueError, match="Each observation slice must be a 1D tensor/array."):
        net.step(obs_slices_3d, action, reward=1.0)


def test_vulnerability_2_direct_flat_lists():
    """
    Vulnerability 2: Passing a direct flat list of floats instead of a nested structure.
    Iterating over the flat list would process scalar elements (0-D tensors),
    causing a PyTorch TypeError on len() check.
    """
    cfg = CerebrumConfig(dims=(4, 8), n_settle=2, seed=42)
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg)
    action = Exogenous(np.array([0.1, 0.1]))
    
    # Flat 1D list passed directly
    obs_slices_flat = [1.0, 2.0]
    with pytest.raises(ValueError, match="Each observation slice must be a 1D tensor/array."):
        net.step(obs_slices_flat, action, reward=1.0)


def test_vulnerability_3_module_count_mismatch():
    """
    Vulnerability 3: Passing obs_slices with count mismatching n_modules,
    originally leading to IndexError inside settle loops.
    """
    cfg = CerebrumConfig(dims=(4, 8), n_settle=2, seed=42)
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg)
    action = Exogenous(np.array([0.1, 0.1]))
    
    # Fewer slices (1) than modules (2)
    with pytest.raises(ValueError, match="Number of observation slices.*must match n_modules"):
        net.step([[1.0, 1.0, 1.0, 1.0]], action, reward=1.0)

    # More slices (3) than modules (2)
    with pytest.raises(ValueError, match="Number of observation slices.*must match n_modules"):
        net.step([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], action, reward=1.0)


def test_other_adversarial_types():
    """
    Test other invalid types and objects inside observation slices.
    """
    cfg = CerebrumConfig(dims=(4, 8), n_settle=2, seed=42)
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg)
    action = Exogenous(np.array([0.1, 0.1]))

    # String in slice
    with pytest.raises(TypeError, match="Observations must be numeric."):
        net.step([["a", "b", "c", "d"], [1.0, 1.0, 1.0, 1.0]], action, reward=1.0)

    # Dict in slice
    with pytest.raises(TypeError, match="Observations must be numeric."):
        net.step([[{"key": 1.0}, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], action, reward=1.0)

    # Custom object in slice
    with pytest.raises(TypeError, match="Observations must be numeric."):
        net.step([[object(), 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], action, reward=1.0)

    # Invalid outer type (e.g. dict)
    with pytest.raises(TypeError, match="obs_slices must be a list or tuple of slices."):
        net.step({0: [1.0, 1.0, 1.0, 1.0], 1: [1.0, 1.0, 1.0, 1.0]}, action, reward=1.0)

    # Invalid outer type (e.g. string)
    with pytest.raises(TypeError, match="obs_slices must be a list or tuple of slices."):
        net.step("invalid", action, reward=1.0)


def test_none_sanitization():
    """
    Verify that None inside a slice is converted/sanitized successfully to 0.0 (nan-like)
    without crashing.
    """
    cfg = CerebrumConfig(dims=(4, 8), n_settle=2, seed=42)
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg)
    action = Exogenous(np.array([0.1, 0.1]))

    obs_slices = [[None, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]
    z, M = net.step(obs_slices, action, reward=1.0)
    assert M is not None
    assert z is not None
