import numpy as np
import torch
from dataclasses import dataclass

# Monkeypatch torch.Tensor for NumPy compatibility
if not hasattr(torch.Tensor, 'copy'):
    torch.Tensor.copy = lambda self: self.clone()

# compat sum
_orig_sum = torch.Tensor.sum
def _compat_sum(self, *args, **kwargs):
    dim = kwargs.pop('axis', None)
    if dim is not None:
        kwargs['dim'] = dim
    out = kwargs.pop('out', None)
    if out is not None:
        kwargs['out'] = out
    keepdim = kwargs.pop('keepdims', None)
    if keepdim is not None:
        kwargs['keepdim'] = keepdim
    return _orig_sum(self, *args, **kwargs)
torch.Tensor.sum = _compat_sum

# compat all
_orig_all = torch.Tensor.all
def _compat_all(self, *args, **kwargs):
    dim = kwargs.pop('axis', None)
    if dim is not None:
        kwargs['dim'] = dim
    out = kwargs.pop('out', None)
    if out is not None:
        kwargs['out'] = out
    keepdim = kwargs.pop('keepdims', None)
    if keepdim is not None:
        kwargs['keepdim'] = keepdim
    return _orig_all(self, *args, **kwargs)
torch.Tensor.all = _compat_all

# compat any
_orig_any = torch.Tensor.any
def _compat_any(self, *args, **kwargs):
    dim = kwargs.pop('axis', None)
    if dim is not None:
        kwargs['dim'] = dim
    out = kwargs.pop('out', None)
    if out is not None:
        kwargs['out'] = out
    keepdim = kwargs.pop('keepdims', None)
    if keepdim is not None:
        kwargs['keepdim'] = keepdim
    return _orig_any(self, *args, **kwargs)
torch.Tensor.any = _compat_any

# compat mean
_orig_mean = torch.Tensor.mean
def _compat_mean(self, *args, **kwargs):
    if not (torch.is_floating_point(self) or torch.is_complex(self)):
        self = self.to(torch.float64)
    dim = kwargs.pop('axis', None)
    if dim is not None:
        kwargs['dim'] = dim
    out = kwargs.pop('out', None)
    if out is not None:
        kwargs['out'] = out
    keepdim = kwargs.pop('keepdims', None)
    if keepdim is not None:
        kwargs['keepdim'] = keepdim
    return _orig_mean(self, *args, **kwargs)
torch.Tensor.mean = _compat_mean

@dataclass(frozen=True)
class Exogenous:
    """An action/motor signal that is, by construction, NOT a function of network state.
    Only values explicitly wrapped here (from the task/environment) can drive the grid
    transition. This makes a data-dependent z_act a type error (BAN-1)."""
    value: object
    def __post_init__(self):
        v = self.value
        if isinstance(v, torch.Tensor):
            pass
        elif isinstance(v, np.ndarray):
            pass
        elif isinstance(v, TensorSliceWrapper):
            v = v._tensor
        else:
            v = np.asarray(v, dtype=float)
        object.__setattr__(self, "value", v)

def to_torch_dtype(dt):
    if isinstance(dt, torch.dtype):
        return dt
    if dt is None:
        return torch.float64
    dt_str = str(dt)
    if 'float32' in dt_str:
        return torch.float32
    if 'float64' in dt_str or 'float' in dt_str:
        return torch.float64
    if 'int32' in dt_str:
        return torch.int32
    if 'int64' in dt_str or 'int' in dt_str:
        return torch.int64
    if 'bool' in dt_str:
        return torch.bool
    return torch.float64

# Monkeypatch astype compatibility
if not hasattr(torch.Tensor, 'astype'):
    torch.Tensor.astype = lambda self, dtype: self.detach().cpu().numpy().astype(dtype)

def safe_to(tensor, device, dtype):
    dev_obj = torch.device(device)
    if dev_obj.type == 'mps':
        return tensor.to(dtype=dtype).to(device=dev_obj)
    else:
        return tensor.to(device=dev_obj).to(dtype=dtype)

def to_tensor(x, device, dtype):
    if x is None:
        return None
    dtype = to_torch_dtype(dtype)
    if isinstance(x, TensorSliceWrapper):
        return safe_to(x._tensor, device, dtype)
    if isinstance(x, torch.Tensor):
        return safe_to(x, device, dtype)
    if isinstance(x, (list, tuple)):
        return [to_tensor(item, device, dtype) for item in x]
    if isinstance(x, np.ndarray):
        return safe_to(torch.from_numpy(x), device, dtype)
    if isinstance(x, (int, float)):
        return safe_to(torch.tensor(x), device, dtype)
    return safe_to(torch.tensor(x), device, dtype)


class TensorSliceWrapper:
    def __init__(self, tensor, device, dtype):
        self._tensor = tensor
        self._device = device
        self._dtype = dtype

    @property
    def device(self):
        return self._tensor.device

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        def unwrap(x):
            if isinstance(x, TensorSliceWrapper):
                return x._tensor
            if isinstance(x, (list, tuple)):
                return type(x)(unwrap(item) for item in x)
            return x
        unwrapped_args = tuple(unwrap(a) for a in args)
        unwrapped_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        return func(*unwrapped_args, **unwrapped_kwargs)

    def __getattr__(self, name):
        return getattr(self._tensor, name)

    def __setitem__(self, idx, value):
        with torch.no_grad():
            self._tensor[idx] = to_tensor(value, self._device, self._dtype)

    def __getitem__(self, idx):
        res = self._tensor[idx]
        if isinstance(res, torch.Tensor):
            return TensorSliceWrapper(res, self._device, self._dtype)
        return res

    def __add__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return self._tensor + other_t

    def __radd__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return other_t + self._tensor

    def __iadd__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        with torch.no_grad():
            self._tensor.copy_(self._tensor + other_t)
        return self

    def __sub__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return self._tensor - other_t

    def __rsub__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return other_t - self._tensor

    def __isub__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        with torch.no_grad():
            self._tensor.copy_(self._tensor - other_t)
        return self

    def __mul__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return self._tensor * other_t

    def __rmul__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return other_t * self._tensor

    def __imul__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        with torch.no_grad():
            self._tensor.copy_(self._tensor * other_t)
        return self

    def __truediv__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return self._tensor / other_t

    def __rtruediv__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return other_t / self._tensor

    def __itruediv__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        with torch.no_grad():
            self._tensor.copy_(self._tensor / other_t)
        return self

    def __matmul__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return self._tensor @ other_t

    def __rmatmul__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return other_t @ self._tensor

    def __pow__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return self._tensor ** other_t

    def __rpow__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return other_t ** self._tensor

    def __neg__(self):
        return -self._tensor

    def __pos__(self):
        return +self._tensor

    def __abs__(self):
        return torch.abs(self._tensor)

    def __lt__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return self._tensor < other_t

    def __le__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return self._tensor <= other_t

    def __gt__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return self._tensor > other_t

    def __ge__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return self._tensor >= other_t

    def __eq__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return self._tensor == other_t

    def __ne__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return self._tensor != other_t

    def __len__(self):
        return len(self._tensor)

    def __iter__(self):
        return iter(TensorSliceWrapper(t, self._device, self._dtype) if isinstance(t, torch.Tensor) else t for t in self._tensor)

    def __repr__(self):
        return f"TensorSliceWrapper({repr(self._tensor)})"

    def __str__(self):
        return str(self._tensor)

    def clone(self):
        return TensorSliceWrapper(self._tensor.clone(), self._device, self._dtype)

    def copy(self):
        return TensorSliceWrapper(self._tensor.clone(), self._device, self._dtype)


    @property
    def ndim(self):
        return self._tensor.ndim

    @property
    def shape(self):
        return self._tensor.shape

    @property
    def dtype(self):
        return self._tensor.dtype

    @property
    def size(self):
        return self._tensor.numel()

    def numel(self):
        return self._tensor.numel()

    def to(self, device, dtype=None):
        dt = dtype if dtype is not None else self._dtype
        return TensorSliceWrapper(safe_to(self._tensor, device, dt), device, dt)

    def __array__(self, dtype=None, copy=None):
        arr = self._tensor.detach().cpu().numpy()
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr


class PyTorchListWrapper:
    def __init__(self, tensors, device, dtype):
        self._tensors = [to_tensor(t, device, dtype) for t in tensors]
        self.device = device
        self.dtype = dtype

    def __len__(self):
        return len(self._tensors)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [TensorSliceWrapper(t, self.device, self.dtype) for t in self._tensors[idx]]
        res = self._tensors[idx]
        if isinstance(res, torch.Tensor):
            return TensorSliceWrapper(res, self.device, self.dtype)
        return res

    def __setitem__(self, idx, value):
        if isinstance(idx, slice):
            self._tensors[idx] = [to_tensor(item, self.device, self.dtype) for item in value]
        else:
            self._tensors[idx] = to_tensor(value, self.device, self.dtype)

    def __iter__(self):
        return iter(TensorSliceWrapper(t, self.device, self.dtype) if isinstance(t, torch.Tensor) else t for t in self._tensors)

    def __repr__(self):
        return repr(self._tensors)

    def append(self, value):
        self._tensors.append(to_tensor(value, self.device, self.dtype))

    def copy(self):
        return list(self._tensors)

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        # Update self._tensors list and wrap in correct device/dtype
        self._tensors = [to_tensor(t, device, self.dtype) for t in self._tensors]
        return self
