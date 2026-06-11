import numpy as np
import torch
from .types import Exogenous

def to_numpy(x):
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def assert_one_hot(z, axis=0, tol=1e-9):
    """Each slice along `axis` must be one-hot OR all-zero (an unfilled slot). Soft weights raise (BAN-1)."""
    z_np = to_numpy(z).astype(float)
    moved = np.moveaxis(z_np, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        nz = col[np.abs(col) > tol]
        assert nz.size <= 1, f"slot {j} has {nz.size} nonzeros -> soft mixing, BAN-1 violation"
        if nz.size == 1:
            assert abs(nz[0] - 1.0) < 1e-6, f"slot {j} nonzero={nz[0]} != 1.0 -> not one-hot, BAN-1"

def assert_scalar_M(M):
    """Neuromodulator must be a scalar; a vector global signal is DFA (BAN-2)."""
    arr = to_numpy(M)
    assert arr.ndim == 0 or arr.size == 1, f"neuromodulator must be scalar, got shape {arr.shape} (DFA/BAN-2)"

def assert_exogenous_action(action):
    """Grid transition driver must be Exogenous (BAN-1: z_act not data-dependent)."""
    if not isinstance(action, Exogenous):
        raise TypeError("z_act must be Exogenous(...) — a data-dependent action is BAN-1 (selective-SSM)")
