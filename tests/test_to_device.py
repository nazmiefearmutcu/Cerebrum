import numpy as np
import pytest
import torch

from cerebrum.config import CerebrumConfig
from cerebrum.core_net import CerebrumCore
from cerebrum.workspace_net import CerebrumWorkspaceNet
from cerebrum.unified import CerebrumNet
from cerebrum.types import Exogenous, PyTorchListWrapper, TensorSliceWrapper


def assert_all_tensors_on_device_and_dtype(obj, device, dtype, exclude_paths=None):
    """
    Recursively crawls the object graph of any network or submodule,
    locating and verifying that all torch.Tensors, TensorSliceWrappers,
    and PyTorchListWrappers are on the specified device and dtype.
    """
    if exclude_paths is None:
        exclude_paths = set()
    target_device = torch.device(device)
    visited = set()
    
    def check(val, path=""):
        if val is None:
            return
        if id(val) in visited:
            return
        visited.add(id(val))
        
        if path in exclude_paths:
            return
            
        if isinstance(val, torch.Tensor):
            assert val.device.type == target_device.type, (
                f"Tensor at '{path}' device type mismatch: actual={val.device.type}, "
                f"expected={target_device.type} (tensor={val.device}, target={target_device})"
            )
            if target_device.index is not None and val.device.index is not None:
                assert val.device.index == target_device.index, (
                    f"Tensor at '{path}' device index mismatch: actual={val.device.index}, "
                    f"expected={target_device.index}"
                )
            assert val.dtype == dtype, (
                f"Tensor at '{path}' dtype mismatch: actual={val.dtype}, expected={dtype}"
            )
        elif isinstance(val, TensorSliceWrapper):
            # Check wrapper attributes
            actual_w_device = torch.device(val._device)
            assert actual_w_device.type == target_device.type, (
                f"TensorSliceWrapper at '{path}' device mismatch: actual={val._device}, expected={device}"
            )
            assert val._dtype == dtype, (
                f"TensorSliceWrapper at '{path}' dtype mismatch: actual={val._dtype}, expected={dtype}"
            )
            # Check inner tensor
            check(val._tensor, path + "._tensor")
        elif isinstance(val, PyTorchListWrapper):
            actual_w_device = torch.device(val.device)
            assert actual_w_device.type == target_device.type, (
                f"PyTorchListWrapper at '{path}' device mismatch: actual={val.device}, expected={device}"
            )
            assert val.dtype == dtype, (
                f"PyTorchListWrapper at '{path}' dtype mismatch: actual={val.dtype}, expected={dtype}"
            )
            for i, t in enumerate(val._tensors):
                check(t, f"{path}._tensors[{i}]")
        elif isinstance(val, (list, tuple)):
            for i, item in enumerate(val):
                check(item, f"{path}[{i}]")
        elif isinstance(val, dict):
            for k, v in val.items():
                check(v, f"{path}['{k}']")
        elif hasattr(val, "__dict__"):
            for k, v in val.__dict__.items():
                if k in ("cfg", "counters"):
                    continue
                check(v, f"{path}.{k}")

    check(obj, "root")


def test_to_method_casting_core_variables_succeeds():
    """Verify that all core variables (excluding known hook bugs) are cast correctly."""
    cfg = CerebrumConfig(dims=(4, 8), seed=42)
    net = CerebrumNet(n_modules=2, k_slots=2, slice_dim=4, cfg=cfg, device='cpu', dtype=torch.float64)
    net.set_backend("torch", device="cpu")
    
    # Run a step
    rng = np.random.default_rng(42)
    obs = [rng.standard_normal(4) * 0.4 for _ in range(2)]
    action = Exogenous(np.array([0.1, -0.2]))
    net.step(obs, action, reward=1.0)
    
    # Exclude last_theta hooks from core verification
    excludes = {
        "root.last_theta",
        "root.last_theta[0]",
        "root.last_theta[0][0]",
        "root.last_theta[1]",
        "root.last_theta[1][0]"
    }
    
    # 1. Cast to CPU float32
    net.to('cpu', torch.float32)
    assert_all_tensors_on_device_and_dtype(net, 'cpu', torch.float32, exclude_paths=excludes)
    
    # 2. Cast to MPS float32 if available
    if torch.backends.mps.is_available():
        net.to('mps', torch.float32)
        assert_all_tensors_on_device_and_dtype(net, 'mps', torch.float32, exclude_paths=excludes)


def test_last_theta_casts_correctly():
    """Verify that CerebrumNet's last_theta list is updated during casting."""
    cfg = CerebrumConfig(dims=(4, 8), seed=42)
    net = CerebrumNet(n_modules=2, k_slots=2, slice_dim=4, cfg=cfg, device='cpu', dtype=torch.float64)
    net.set_backend("torch", device="cpu")
    
    rng = np.random.default_rng(42)
    obs = [rng.standard_normal(4) * 0.4 for _ in range(2)]
    action = Exogenous(np.array([0.1, -0.2]))
    net.step(obs, action, reward=1.0)
    
    net.to('cpu', torch.float32)
    assert net.last_theta[0][0].dtype == torch.float32


def test_mps_to_cpu_float64_casting_succeeds():
    """Verify that casting from MPS to CPU float64 succeeds due to two-step casting implementation."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
        
    cfg = CerebrumConfig(dims=(6, 5, 4), seed=42)
    net = CerebrumCore(cfg, device='cpu', dtype=torch.float64)
    
    # Settle state on MPS float32
    net.to('mps', torch.float32)
    
    # Try to cast back to CPU float64. This should succeed now.
    net.to('cpu', torch.float64)


def test_to_method_casting_core_net_without_mps_to_cpu_bug():
    """Verify that CerebrumCore correctly casts its sub-tensors on .to() without triggering the MPS back-casting bug."""
    cfg = CerebrumConfig(dims=(6, 5, 4), seed=42)
    net = CerebrumCore(cfg, device='cpu', dtype=torch.float64)
    obs = np.array([0.2, -0.1, 0.3, 0.0, 0.5, -0.2])
    net.observe_and_learn(obs, reward=1.0)

    assert_all_tensors_on_device_and_dtype(net, 'cpu', torch.float64)

    # Cast to CPU float32
    net.to('cpu', torch.float32)
    assert_all_tensors_on_device_and_dtype(net, 'cpu', torch.float32)

    # Cast to MPS float32 if available
    if torch.backends.mps.is_available():
        net.to('mps', torch.float32)
        assert_all_tensors_on_device_and_dtype(net, 'mps', torch.float32)


def test_to_method_casting_workspace_net_without_mps_to_cpu_bug():
    """Verify that CerebrumWorkspaceNet correctly casts its sub-tensors on .to() without triggering the MPS back-casting bug."""
    cfg = CerebrumConfig(dims=(4, 8), seed=42)
    net = CerebrumWorkspaceNet(n_modules=2, k_slots=2, slice_dim=4, cfg=cfg, device='cpu', dtype=torch.float64)
    rng = np.random.default_rng(42)
    obs = [rng.standard_normal(4) * 0.4 for _ in range(2)]
    net.step(obs, reward=1.0)

    assert_all_tensors_on_device_and_dtype(net, 'cpu', torch.float64)

    # Cast to CPU float32
    net.to('cpu', torch.float32)
    assert_all_tensors_on_device_and_dtype(net, 'cpu', torch.float32)

    # Cast to MPS float32 if available
    if torch.backends.mps.is_available():
        net.to('mps', torch.float32)
        assert_all_tensors_on_device_and_dtype(net, 'mps', torch.float32)


def test_mathematical_equivalence_cpu_vs_mps():
    """Verify mathematical equivalence of PyTorch backend execution on CPU vs MPS."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available on this system.")

    cfg_cpu = CerebrumConfig(dims=(4, 8), seed=42)
    cfg_mps = CerebrumConfig(dims=(4, 8), seed=42)

    # Create network on CPU and switch to torch backend (float32)
    net_cpu = CerebrumNet(n_modules=2, k_slots=2, slice_dim=4, cfg=cfg_cpu)
    net_cpu.set_backend("torch", device="cpu")
    net_cpu.to("cpu", torch.float32)

    # Create network on MPS and switch to torch backend (float32)
    net_mps = CerebrumNet(n_modules=2, k_slots=2, slice_dim=4, cfg=cfg_mps)
    net_mps.set_backend("torch", device="mps")

    # Generate identical inputs
    rng = np.random.default_rng(12345)
    
    # We will run 5 identical steps on both networks and compare outputs, weights, and internal states
    for step_idx in range(5):
        obs = [rng.standard_normal(4) * 0.4 for _ in range(2)]
        action = Exogenous(np.array([0.15 * (step_idx + 1), -0.1 * (step_idx + 1)]))
        reward = 1.0 + 0.5 * step_idx

        # Run CPU
        z_cpu, M_cpu = net_cpu.step(obs, action, reward)

        # Run MPS
        z_mps, M_mps = net_mps.step(obs, action, reward)

        # Compare outputs: convert MPS outputs to CPU first
        z_cpu_np = z_cpu.cpu().numpy() if isinstance(z_cpu, torch.Tensor) else np.asarray(z_cpu)
        z_mps_np = z_mps.cpu().numpy() if isinstance(z_mps, torch.Tensor) else np.asarray(z_mps)
        assert np.array_equal(z_cpu_np, z_mps_np), f"Step {step_idx}: Routing z mismatch!"

        M_cpu_val = float(M_cpu.item()) if isinstance(M_cpu, torch.Tensor) else float(M_cpu)
        M_mps_val = float(M_mps.item()) if isinstance(M_mps, torch.Tensor) else float(M_mps)
        assert M_cpu_val == pytest.approx(M_mps_val, rel=1e-5, abs=1e-5), f"Step {step_idx}: Neuromodulator M mismatch!"

        # Compare internal states
        slots_cpu = net_cpu.workspace.slots.cpu().numpy()
        slots_mps = net_mps.workspace.slots.cpu().numpy()
        assert np.allclose(slots_cpu, slots_mps, atol=1e-5, rtol=1e-5), f"Step {step_idx}: Workspace slots mismatch!"

        pos_cpu = net_cpu.grid.pos.cpu().numpy()
        pos_mps = net_mps.grid.pos.cpu().numpy()
        assert np.allclose(pos_cpu, pos_mps, atol=1e-5, rtol=1e-5), f"Step {step_idx}: Grid pos mismatch!"

        if net_cpu.grid.store is not None:
            store_cpu = net_cpu.grid.store.cpu().numpy()
            store_mps = net_mps.grid.store.cpu().numpy()
            assert np.allclose(store_cpu, store_mps, atol=1e-5, rtol=1e-5), f"Step {step_idx}: Grid store mismatch!"

        # PCAreas weights and activations
        for m_idx in range(2):
            mod_cpu = net_cpu.modules[m_idx]
            mod_mps = net_mps.modules[m_idx]
            for l in range(mod_cpu.L - 1):
                W_cpu = mod_cpu.W[l].cpu().numpy()
                W_mps = mod_mps.W[l].cpu().numpy()
                assert np.allclose(W_cpu, W_mps, atol=1e-5, rtol=1e-5), f"Step {step_idx}: Module {m_idx} layer {l} W mismatch!"

                B_cpu = mod_cpu.B[l].cpu().numpy()
                B_mps = mod_mps.B[l].cpu().numpy()
                assert np.allclose(B_cpu, B_mps, atol=1e-5, rtol=1e-5), f"Step {step_idx}: Module {m_idx} layer {l} B mismatch!"

                Pi_cpu = mod_cpu.Pi[l].cpu().numpy()
                Pi_mps = mod_mps.Pi[l].cpu().numpy()
                assert np.allclose(Pi_cpu, Pi_mps, atol=1e-5, rtol=1e-5), f"Step {step_idx}: Module {m_idx} layer {l} Pi mismatch!"
