import numpy as np
import torch

class SeededRNG:
    """Centralized reproducible noise.
    Can run in both NumPy parity mode (CPU-based) for tests,
    and native PyTorch mode for production.
    """
    def __init__(self, seed: int = 0, enabled: bool = True, device='cpu', dtype=torch.float64, mode='parity'):
        self.seed = seed
        self.enabled = enabled
        self.device = device
        self.dtype = dtype
        self.mode = mode
        
        # NumPy parity generator
        self._rng = np.random.default_rng(seed)
        
        # PyTorch native generator (on CPU for safety/reproducibility across CUDA/MPS/CPU)
        self._generator = torch.Generator(device='cpu')
        self._generator.manual_seed(seed)

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    def normal(self, shape, scale: float = 1.0):
        if not self.enabled:
            return torch.zeros(shape, device=self.device, dtype=self.dtype)
        
        if self.mode == 'parity':
            val = self._rng.normal(0.0, scale, size=shape)
            # Support both float and array cases, ensuring it's a tensor on the correct device/dtype
            return torch.tensor(val, device=self.device, dtype=self.dtype)
        else:
            # Native PyTorch mode
            val = torch.randn(shape, generator=self._generator, dtype=self.dtype).to(self.device)
            return val * scale

    def gumbel(self, shape):
        if not self.enabled:
            return torch.zeros(shape, device=self.device, dtype=self.dtype)
        
        if self.mode == 'parity':
            val = self._rng.gumbel(0.0, 1.0, size=shape)
            return torch.tensor(val, device=self.device, dtype=self.dtype)
        else:
            # Native PyTorch mode using inverse transform sampling: G = -ln(-ln(U))
            U = torch.rand(shape, generator=self._generator, dtype=self.dtype)
            val = -torch.log(-torch.log(U + 1e-20) + 1e-20)
            return val.to(self.device)

    def uniform(self, shape):
        if not self.enabled:
            return torch.zeros(shape, device=self.device, dtype=self.dtype)
        
        if self.mode == 'parity':
            val = self._rng.uniform(0.0, 1.0, size=shape)
            return torch.tensor(val, device=self.device, dtype=self.dtype)
        else:
            # Native PyTorch mode
            val = torch.rand(shape, generator=self._generator, dtype=self.dtype)
            return val.to(self.device)
