import pytest
import torch
import numpy as np
from cerebrum.config import CerebrumConfig
from cerebrum.unified import CerebrumNet
from cerebrum.types import Exogenous, TensorSliceWrapper, PyTorchListWrapper

def check_net_device_and_dtype(net, device, dtype):
    # Normalize device string
    device_type = torch.device(device).type

    # Check top-level properties
    assert net.device == device
    assert net.dtype == dtype
    if net._U is not None:
        assert isinstance(net._U, torch.Tensor)
        assert net._U.device.type == device_type
        assert net._U.dtype == dtype
    if isinstance(net.last_top_pred, torch.Tensor):
        assert net.last_top_pred.device.type == device_type
        assert net.last_top_pred.dtype == dtype

    # Check modules
    for mod in net.modules:
        assert mod.device == device
        assert mod.dtype == dtype
        
        # Check inner variables in PyTorchListWrapper
        assert isinstance(mod.x, PyTorchListWrapper)
        assert mod.x.device == device
        assert mod.x.dtype == dtype
        for t in mod.x._tensors:
            assert isinstance(t, torch.Tensor)
            assert t.device.type == device_type
            assert t.dtype == dtype

        assert isinstance(mod.eps, PyTorchListWrapper)
        assert mod.eps.device == device
        assert mod.eps.dtype == dtype
        for t in mod.eps._tensors:
            assert isinstance(t, torch.Tensor)
            assert t.device.type == device_type
            assert t.dtype == dtype

        assert isinstance(mod.Pi, PyTorchListWrapper)
        assert mod.Pi.device == device
        assert mod.Pi.dtype == dtype
        for t in mod.Pi._tensors:
            assert isinstance(t, torch.Tensor)
            assert t.device.type == device_type
            assert t.dtype == dtype

        assert isinstance(mod.W, PyTorchListWrapper)
        assert mod.W.device == device
        assert mod.W.dtype == dtype
        for t in mod.W._tensors:
            assert isinstance(t, torch.Tensor)
            assert t.device.type == device_type
            assert t.dtype == dtype

        assert isinstance(mod.B, PyTorchListWrapper)
        assert mod.B.device == device
        assert mod.B.dtype == dtype
        for t in mod.B._tensors:
            assert isinstance(t, torch.Tensor)
            assert t.device.type == device_type
            assert t.dtype == dtype

    # Check grid
    assert net.grid.device == device
    assert net.grid.dtype == dtype
    assert isinstance(net.grid.k, torch.Tensor)
    assert net.grid.k.device.type == device_type
    assert net.grid.k.dtype == dtype
    assert isinstance(net.grid.pos, torch.Tensor)
    assert net.grid.pos.device.type == device_type
    assert net.grid.pos.dtype == dtype
    if net.grid.store is not None:
        assert isinstance(net.grid.store, torch.Tensor)
        assert net.grid.store.device.type == device_type
        assert net.grid.store.dtype == dtype

    # Check gate
    assert net.gate.device == device
    assert net.gate.dtype == dtype
    assert isinstance(net.gate.G, torch.Tensor)
    assert net.gate.G.device.type == device_type
    assert net.gate.G.dtype == dtype
    assert isinstance(net.gate.N, torch.Tensor)
    assert net.gate.N.device.type == device_type
    assert net.gate.N.dtype == dtype
    assert isinstance(net.gate.theta, torch.Tensor)
    assert net.gate.theta.device.type == device_type
    assert net.gate.theta.dtype == dtype
    if net.gate._P is not None:
        assert isinstance(net.gate._P, torch.Tensor)
        assert net.gate._P.device.type == device_type
        assert net.gate._P.dtype == dtype
    if net.gate._z is not None:
        assert isinstance(net.gate._z, torch.Tensor)
        assert net.gate._z.device.type == device_type
        assert net.gate._z.dtype == dtype
    if net.gate._bid is not None:
        assert isinstance(net.gate._bid, torch.Tensor)
        assert net.gate._bid.device.type == device_type
        assert net.gate._bid.dtype == dtype

    # Check workspace
    assert net.workspace.device == device
    assert net.workspace.dtype == dtype
    assert isinstance(net.workspace.slots, torch.Tensor)
    assert net.workspace.slots.device.type == device_type
    assert net.workspace.slots.dtype == dtype

    # Check neuromodulator
    assert net.nm.device == device
    assert net.nm.dtype == dtype

    # Check SeededRNG
    assert net.rng.device == device
    assert net.rng.dtype == dtype

    # Check eligibility traces
    for m_elig in net.elig:
        for e in m_elig:
            assert e.device == device
            assert e.dtype == dtype
            assert isinstance(e.value, torch.Tensor)
            assert e.value.device.type == device_type
            assert e.value.dtype == dtype

    # Check metaplastic fuses
    for m_fuse in net.fuse:
        for f in m_fuse:
            assert f.device == device
            assert f.dtype == dtype
            assert isinstance(f.c, torch.Tensor)
            assert f.c.device.type == device_type
            assert f.c.dtype == dtype
            assert isinstance(f.S_bar, torch.Tensor)
            assert f.S_bar.device.type == device_type
            assert f.S_bar.dtype == dtype

@pytest.mark.e2e
def test_to_method_casting():
    cfg = CerebrumConfig(dims=(4, 8), n_settle=8, seed=42)
    net = CerebrumNet(n_modules=2, k_slots=1, slice_dim=4, cfg=cfg)
    
    # Initialize some optional tensors by running a step first on CPU
    net.set_backend("torch", device="cpu")
    obs = [np.random.randn(4) for _ in range(2)]
    action = Exogenous(np.array([0.1, -0.1]))
    
    # Run initial step
    net.step(obs, action, reward=1.0)
    
    # Check that initial CPU state is correct (default dtype is float64)
    check_net_device_and_dtype(net, "cpu", torch.float64)
    
    # Cast to float32 on CPU
    net.to("cpu", dtype=torch.float32)
    check_net_device_and_dtype(net, "cpu", torch.float32)
    
    # Run a step on CPU float32
    net.step(obs, action, reward=1.0)
    
    # If MPS is available, test casting to MPS
    if torch.backends.mps.is_available():
        # Cast to MPS float32
        net.to("mps", dtype=torch.float32)
        check_net_device_and_dtype(net, "mps", torch.float32)
        
        # Run a step on MPS float32
        net.step(obs, action, reward=1.0)
        
        # Cast back to CPU float32 (should succeed)
        net.to("cpu", dtype=torch.float32)
        check_net_device_and_dtype(net, "cpu", torch.float32)
        
        # Run a step on CPU float32
        net.step(obs, action, reward=1.0)
        
        # Attempt to cast from MPS to CPU float64.
        # This used to trigger a TypeError because PyTorch's MPS backend raises an error
        # when a double/float64 casting is requested directly on an MPS tensor.
        # With the two-step casting fix, it should succeed.
        net.to("mps", dtype=torch.float32)
        net.to("cpu", dtype=torch.float64)
        check_net_device_and_dtype(net, "cpu", torch.float64)
