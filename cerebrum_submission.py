__version__ = "0.0.1"

import numpy as np
import torch
from dataclasses import dataclass, field, replace


# ==========================================
# cerebrum/config.py
# ==========================================


@dataclass(frozen=True)
class CerebrumConfig:
    # PC hierarchy
    dims: tuple = (16, 12, 8)        # area sizes, dims[0] = observation/lowest area
    # timescales (smaller = faster)
    tau_x: float = 1.0               # activity settling
    tau_e: float = 5.0               # eligibility trace
    tau_w: float = 200.0             # weight plasticity (slow)
    tau_pi: float = 100.0            # precision learning
    tau_r: float = 50.0              # reward baseline EMA
    tau_b: float = 200.0             # feedback weight learning
    # settling
    dt: float = 0.1
    n_settle: int = 40
    T_floor: float = 0.02            # Pillar 4: noise floor > 0 forbids MAP collapse
    T0: float = 0.2                  # initial annealing temperature
    tau_anneal: float = 15.0
    # costs / rates
    gamma: float = 0.01              # activity L1 sparsity (R(x))
    eta_w: float = 0.02              # weight learning rate scale
    eta_b: float = 0.01              # feedback weight learning rate
    lam_b: float = 1e-3              # feedback weight decay
    # --- Kolen-Pollack feedback alignment (OPT-IN; default OFF = behavior unchanged) ---
    align_feedback: bool = False     # if True, B is driven toward W.T by a matched LOCAL rule
                                     # (KP): B and W receive the SAME M-gated pre*post product
                                     # (transposed) with a MATCHED decay, so (W - B.T) -> 0
                                     # WITHOUT ever reading/copying W.T (no weight transport).
    lam_kp: float = 1e-2             # matched symmetric weight decay applied to BOTH W and B
                                     # ONLY when align_feedback=True (the KP coupling term).
    # --- top-down precision balancing (OPT-IN; default OFF = behavior unchanged) ---
    balance_grid_precision: bool = False
                                     # if True, the EXTERNAL top-down prediction at the TOP area
                                     # (the grid HEAD's structural prediction in CerebrumNet/CerebrumCore)
                                     # is gain-normalized to the bottom-up activity scale of that
                                     # area BEFORE it enters the top-area error. In predictive coding
                                     # the relative pull of a top-down prediction is set by PRECISION;
                                     # the never-decayed Hebbian grid content store makes ||top_pred||
                                     # ~500x the small obs-driven latent (|x|~0.1), so its unit-precision
                                     # pull CRUSHES the obs factor code (the latent tracks grid phase,
                                     # not obs factors). This LOCAL, per-area diagonal gain rescales
                                     # the prediction so the grid top-down and the bottom-up
                                     # reconstruction signal are weighted COMPARABLY. NO global
                                     # objective, no weight transport — a pure prediction-gain op.
    grid_precision_ref: float = 1.0  # target ratio ||scaled top_pred|| / ||bottom-up activity||;
                                     # 1.0 = match the bottom-up scale exactly (only used when
                                     # balance_grid_precision=True).
    Pi0: float = 1.0                 # precision prior
    sigma0: float = 1.0              # precision floor variance
    kappa_pi: float = 1.0            # precision learning gain
    # grid HEAD
    grid_n_modules: int = 6
    grid_lambda0: float = 4.0        # base spatial period
    grid_ratio: float = 1.42         # geometric module scaling
    grid_eta_bind: float = 1.0       # content-store binding rate
    # gate / workspace (Stage 2)
    lam_g: float = 0.0        # gate Go/NoGo weight decay toward init (0 = off; >0 prevents spurious
                              # preference drift when there is no stable per-module target to learn)
    gate_temp: float = 0.0    # fixed gate selection temperature (0 = unset -> use neuromodulator 1/M);
                              # a low value lets the informative scalar bid dominate (still stochastic)
    # metaplasticity (Stage 3)
    tau_S: float = 20.0       # surprise-baseline EMA timescale
    tau_c: float = 300.0      # consolidation-reserve timescale (slow)
    alpha_c: float = 1.0      # low-surprise consolidation gain (builds c)
    beta_c: float = 1.5       # high-surprise erosion gain (frees c)
    c_max: float = 1.0        # max consolidation reserve
    g_theta: float = 4.0      # plasticity-permission sigmoid sharpness
    # misc
    pc_sparsity_threshold: float = 0.0
    seed: int = 0


# ==========================================
# cerebrum/counters.py
# ==========================================


class Counters:
    def __init__(self):
        self.global_comm_learn = 0    # scalar M events (target O(1))
        self.global_comm_infer = 0    # broadcast vector elements (O(k * T_settle))
        self.synaptic_ops = 0
        self.dense_synaptic_ops = 0
        self.dynamic_synaptic_ops = 0
        self._active = 0; self._total = 0
    def record_global_learn(self, n=1): self.global_comm_learn += n
    def record_global_infer_vectors(self, k, width): self.global_comm_infer += k * width
    def record_synaptic_ops(self, dense, dynamic=None):
        if dynamic is None:
            dynamic = dense
        self.dense_synaptic_ops += int(dense)
        self.dynamic_synaptic_ops += int(dynamic)
        self.synaptic_ops = self.dynamic_synaptic_ops
    def record_activity(self, x, tol=1e-6):
        if hasattr(x, 'detach'):
            self._active += int(torch.sum(torch.abs(x) > tol).item())
            self._total += x.numel()
        else:
            x = np.asarray(x)
            self._active += int(np.sum(np.abs(x) > tol))
            self._total += x.size
    def sparsity(self):  # active fraction rho
        return self._active / self._total if self._total else 0.0
    def reset_activity(self): self._active = 0; self._total = 0


# ==========================================
# cerebrum/types.py
# ==========================================


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

    def __sub__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return self._tensor - other_t

    def __rsub__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return other_t - self._tensor

    def __mul__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return self._tensor * other_t

    def __rmul__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return other_t * self._tensor

    def __truediv__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return self._tensor / other_t

    def __rtruediv__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        other_t = to_tensor(other_t, self._device, self._dtype)
        return other_t / self._tensor

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

    def __lt__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        return self._tensor < other_t

    def __le__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        return self._tensor <= other_t

    def __gt__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        return self._tensor > other_t

    def __ge__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        return self._tensor >= other_t

    def __eq__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        return self._tensor == other_t

    def __ne__(self, other):
        other_t = other._tensor if isinstance(other, TensorSliceWrapper) else other
        return self._tensor != other_t

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


# ==========================================
# cerebrum/invariants.py
# ==========================================


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


# ==========================================
# cerebrum/nonlinear.py
# ==========================================


def g_act(u):
    if isinstance(u, torch.Tensor):
        return torch.tanh(u)
    import numpy as np
    return np.tanh(u)

def g_deriv(u):
    if isinstance(u, torch.Tensor):
        return 1.0 - torch.tanh(u)**2
    import numpy as np
    return 1.0 - np.tanh(u)**2


# ==========================================
# cerebrum/rng.py
# ==========================================


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


# ==========================================
# cerebrum/pc_core.py
# ==========================================


class PCAreas:
    """Hierarchical predictive-coding areas. x[l] predicted from x[l+1] by forward W[l].
    Feedback B[l] is a SEPARATE synapse (no weight transport). Precision Pi[l] is DIAGONAL."""
    def __init__(self, cfg, device='cpu', dtype=torch.float64):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        d = cfg.dims; self.L = len(d)
        rng = np.random.default_rng(cfg.seed)
        
        x_tensors = [torch.zeros(d[l], device=device, dtype=dtype) for l in range(self.L)]
        eps_tensors = [torch.zeros(d[l], device=device, dtype=dtype) for l in range(self.L)]
        Pi_tensors = [torch.full((d[l],), cfg.Pi0, device=device, dtype=dtype) for l in range(self.L)]
        
        self._x = PyTorchListWrapper(x_tensors, device, dtype)
        self._eps = PyTorchListWrapper(eps_tensors, device, dtype)
        self._Pi = PyTorchListWrapper(Pi_tensors, device, dtype)
        
        # W[l]: (d[l], d[l+1]); B[l]: (d[l+1], d[l]) separate feedback (NOT W[l].T)
        # Weight initialization must use NumPy's RNG for seeds
        W_tensors = [torch.tensor(0.1*rng.standard_normal((d[l], d[l+1])), device=device, dtype=dtype) for l in range(self.L-1)]
        B_tensors = [torch.tensor(0.1*rng.standard_normal((d[l+1], d[l])), device=device, dtype=dtype) for l in range(self.L-1)]
        
        self._W = PyTorchListWrapper(W_tensors, device, dtype)
        self._B = PyTorchListWrapper(B_tensors, device, dtype)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, val):
        self._x = PyTorchListWrapper(val, self.device, self.dtype)

    @property
    def eps(self):
        return self._eps

    @eps.setter
    def eps(self, val):
        self._eps = PyTorchListWrapper(val, self.device, self.dtype)

    @property
    def Pi(self):
        return self._Pi

    @Pi.setter
    def Pi(self, val):
        self._Pi = PyTorchListWrapper(val, self.device, self.dtype)

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, val):
        self._W = PyTorchListWrapper(val, self.device, self.dtype)

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, val):
        self._B = PyTorchListWrapper(val, self.device, self.dtype)

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        self._x.to(device, self.dtype)
        self._eps.to(device, self.dtype)
        self._Pi.to(device, self.dtype)
        self._W.to(device, self.dtype)
        self._B.to(device, self.dtype)
        return self


    def _bottomup_scale_top(self):
        """L2 scale of the BOTTOM-UP reconstruction signal driving the TOP area, used as the
        reference against which an external top-down prediction is precision-balanced.
        """
        if self.L < 2:
            return torch.linalg.norm(self.x[-1])
        fprime = g_deriv(self.W[self.L-2] @ self.x[self.L-1])
        fb = self.B[self.L-2] @ (fprime * (self.Pi[self.L-2] * self.eps[self.L-2]))
        s = torch.linalg.norm(fb)
        if s == 0.0:
            s = torch.linalg.norm(self.x[-1])
        return s

    def _balanced_top_pred(self, top_pred):
        """Gain-normalize an EXTERNAL top-down prediction to the top area's bottom-up signal scale.
        OPT-IN via cfg.balance_grid_precision; default OFF returns top_pred unchanged (bit-identical).
        """
        if top_pred is None or not getattr(self.cfg, "balance_grid_precision", False):
            return top_pred
        pnorm = torch.linalg.norm(top_pred)
        if pnorm == 0.0:
            return top_pred
        ref = getattr(self.cfg, "grid_precision_ref", 1.0) * self._bottomup_scale_top()
        scale = torch.minimum(torch.tensor(1.0, device=self.device, dtype=self.dtype), ref / pnorm)
        return top_pred * scale

    def predict(self, l, top_pred=None):
        """top-down prediction of area l."""
        if l < self.L-1:
            return g_act(self.W[l] @ self.x[l+1])
        if top_pred is None:
            return torch.zeros_like(self.x[l])
        top_pred_t = to_tensor(top_pred, self.device, self.dtype)
        return self._balanced_top_pred(top_pred_t)

    def compute_errors(self, top_pred=None, broadcast=None):
        for l in range(self.L):
            yhat = self.predict(l, top_pred=top_pred)
            p = 0.0
            if broadcast is not None:
                p = to_tensor(broadcast[l], self.device, self.dtype)
            self.eps[l] = self.x[l] - yhat - p

    def energy(self):
        e = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        for l in range(self.L):
            e += 0.5*torch.sum(self.Pi[l]*self.eps[l]**2) - 0.5*torch.sum(torch.log(self.Pi[l]))
        return e

    def settle_step(self, rng, T, clamp_bottom=None, top_pred=None, broadcast=None, counters=None):
        self.compute_errors(top_pred=top_pred, broadcast=broadcast)
        c = self.cfg
        new_x = [xl.clone() for xl in self.x]
        
        if isinstance(T, torch.Tensor):
            T_val = float(T.item())
        else:
            T_val = float(T)
            
        for l in range(self.L):
            if l == 0 and clamp_bottom is not None:
                clamp_bottom_t = to_tensor(clamp_bottom, self.device, self.dtype)
                new_x[0] = clamp_bottom_t.clone()
                continue
                
            drift = -self.Pi[l]*self.eps[l]
            if l >= 1:  # feedback from area below via SEPARATE B[l-1] (no transpose of W)
                fprime = g_deriv(self.W[l-1] @ self.x[l])
                drift = drift + self.B[l-1] @ (fprime * (self.Pi[l-1]*self.eps[l-1]))
            drift = drift - c.gamma*torch.sign(self.x[l])     # -dR/dx (L1 sparsity)
            step = (drift/c.tau_x)*c.dt
            noise = rng.normal(self.x[l].shape, scale=np.sqrt(2.0*T_val*c.dt/c.tau_x))
            
            with torch.no_grad():
                new_x[l] = self.x[l] + step + noise
                if l >= 1 and self.cfg.pc_sparsity_threshold > 0.0:
                    new_x[l] = torch.where(torch.abs(new_x[l]) < self.cfg.pc_sparsity_threshold, 
                                           torch.tensor(0.0, device=self.device, dtype=self.dtype), 
                                           new_x[l])
        
        if counters is not None:
            dense_ops = 0
            dyn_ops = 0
            for l in range(self.L - 1):
                d_l = self.cfg.dims[l]
                d_l1 = self.cfg.dims[l+1]
                dense_ops += 2 * d_l * d_l1
                active_x = int(torch.sum(torch.abs(self.x[l+1]) > 1e-6).item())
                dyn_ops += active_x * d_l
                active_eps = int(torch.sum(torch.abs(self.eps[l]) > 1e-6).item())
                dyn_ops += active_eps * d_l1
            counters.record_synaptic_ops(dense_ops, dyn_ops)
            
        self.x = PyTorchListWrapper(new_x, self.device, self.dtype)
        if counters is not None:
            for xl in self.x[1:]:
                counters.record_activity(xl)


# ==========================================
# cerebrum/grid_head.py
# ==========================================


class GridHead:
    """Structured generative prior. Frozen multi-frequency grid modules; phase advanced by
    EXOGENOUS actions only. Code g_m = [cos, sin] of phase. Content store added in Task 11."""
    def __init__(self, cfg, device='cpu', dtype=torch.float64):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        M = cfg.grid_n_modules
        
        # Maintain EXACT seed parity using NumPy RNG for initial parameters
        rng = np.random.default_rng(cfg.seed + 999)
        periods = cfg.grid_lambda0 * (cfg.grid_ratio ** np.arange(M))
        angles = rng.uniform(0, 2*np.pi, size=M)            # frozen orientation per module
        
        k_np = np.stack([(2*np.pi/periods)*np.cos(angles),
                           (2*np.pi/periods)*np.sin(angles)], axis=1)  # (M,2) frozen frequencies
        
        self.k = torch.tensor(k_np, device=device, dtype=dtype)
        self._pos = torch.zeros(2, device=device, dtype=dtype)
        self.store = None
        self.obs_dim = None

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, val):
        self._pos = to_tensor(val, self.device, self.dtype)

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.k = safe_to(self.k, device, self.dtype)
        self._pos = safe_to(self._pos, device, self.dtype)
        if self.store is not None:
            self.store = safe_to(self.store, device, self.dtype)
        return self

    def reset(self):
        self._pos = torch.zeros(2, device=self.device, dtype=self.dtype)

    def transition(self, action):
        assert_exogenous_action(action)
        val = to_tensor(action.value, self.device, self.dtype)
        self._pos = self._pos + val

    def encode(self):
        phase = self.k @ self.pos                            # (M,)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        return torch.stack([cos_phase, sin_phase], dim=1).reshape(-1)

    def _ensure_store(self, obs_dim):
        if self.store is None:
            self.obs_dim = obs_dim
            self.store = torch.zeros((obs_dim, self.encode().numel()), device=self.device, dtype=self.dtype)

    def bind(self, obs, M=1.0):
        obs_t = to_tensor(obs, self.device, self.dtype)
        self._ensure_store(obs_t.numel())
        g = self.encode()
        if isinstance(M, torch.Tensor):
            M_val = safe_to(M, self.device, self.dtype)
        else:
            M_val = float(M)
        with torch.no_grad():
            self.store += self.cfg.grid_eta_bind * M_val * torch.outer(obs_t, g)

    def complete(self):
        g = self.encode()
        if self.store is not None:
            return self.store @ g
        else:
            return torch.zeros(self.obs_dim or 1, device=self.device, dtype=self.dtype)


# ==========================================
# cerebrum/gate.py
# ==========================================


class BasalGangliaGate:
    """Stochastic basal-ganglia gate. Modules bid a SCALAR own-error salience for k workspace slots;
    a striatal Go/NoGo competition selects a strict one-hot winner per slot WITH noise (never argmax,
    never soft). Gate weights learn by a LOCAL three-factor rule gated by the scalar neuromodulator M.
    There is NO query-key / content-similarity term anywhere — the competition can never become attention."""
    def __init__(self, n_modules, k_slots, cfg, seed=0, device='cpu', dtype=torch.float64):
        self.M_ = n_modules
        self.k = k_slots
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        
        rng = np.random.default_rng(seed + 31)
        G_np = 0.5 + 0.1*rng.standard_normal((n_modules, k_slots))   # Go weights
        N_np = 0.1*rng.standard_normal((n_modules, k_slots))         # NoGo weights
        
        self.G = torch.tensor(G_np, device=device, dtype=dtype)
        self.N = torch.tensor(N_np, device=device, dtype=dtype)
        self.theta = torch.zeros(n_modules, device=device, dtype=dtype)  # dead-expert excitability
        self._P = None
        self._z = None
        self._bid = None

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.G = safe_to(self.G, device, self.dtype)
        self.N = safe_to(self.N, device, self.dtype)
        self.theta = safe_to(self.theta, device, self.dtype)
        if self._P is not None:
            self._P = safe_to(self._P, device, self.dtype)
        if self._z is not None:
            self._z = safe_to(self._z, device, self.dtype)
        if self._bid is not None:
            self._bid = safe_to(self._bid, device, self.dtype)
        return self

    def bid(self, err_sq, pi):
        err_sq_t = to_tensor(err_sq, self.device, self.dtype)
        if isinstance(pi, (torch.Tensor, np.ndarray)):
            pi_val = to_tensor(pi, self.device, self.dtype)
        elif isinstance(pi, (list, tuple)):
            pi_val = torch.tensor(pi, device=self.device, dtype=self.dtype)
        else:
            pi_val = float(pi)
        return pi_val * err_sq_t + self.theta


    def select(self, bids, rng, T_gate):
        bids_t = to_tensor(bids, self.device, self.dtype)
        z = torch.zeros((self.M_, self.k), device=self.device, dtype=self.dtype)
        P = torch.zeros((self.M_, self.k), device=self.device, dtype=self.dtype)
        
        if isinstance(T_gate, torch.Tensor):
            T_val = float(T_gate.item())
        else:
            T_val = float(T_gate)
            
        for j in range(self.k):
            inhib_total = torch.sum(self.N[:, j] * bids_t)
            u = self.G[:, j] * bids_t - (inhib_total - self.N[:, j] * bids_t)
            
            # Draw Gumbel noise using SeededRNG
            gumbel_noise = rng.gumbel((self.M_,))
            logits = u / max(T_val, 1e-6) + gumbel_noise
            
            ex = torch.exp(logits - logits.max())
            P[:, j] = ex / ex.sum()
            z[torch.argmax(logits).item(), j] = 1.0
            
        assert_one_hot(z, axis=0)
        self._P, self._z, self._bid = P, z, bids_t
        return z

    def learn(self, M, eta=None):
        assert_scalar_M(M)
        eta_val = self.cfg.eta_w if eta is None else eta
        if isinstance(M, torch.Tensor):
            M_val = safe_to(M, self.device, self.dtype)
        else:
            M_val = float(M)
            
        e = (self._z - self._P) * self._bid[:, None]
        
        with torch.no_grad():
            self.G += eta_val * M_val * e
            self.N += -eta_val * M_val * e
            if self.cfg.lam_g > 0.0:
                self.G += self.cfg.lam_g * (0.5 - self.G)
                self.N += self.cfg.lam_g * (0.0 - self.N)

    def homeostasis(self, M=None, gamma_up=0.02, gamma_dn=0.05):
        wins = torch.minimum(self._z.sum(dim=1), torch.tensor(1.0, device=self.device, dtype=self.dtype))
        if M is None:
            hog = 1.0
        else:
            assert_scalar_M(M)
            if isinstance(M, torch.Tensor):
                M_val = float(M.item())
            else:
                M_val = float(M)
            hog = 1.0 / (1.0 + np.exp(2.0 * M_val))
            
        with torch.no_grad():
            self.theta += gamma_up * (1.0 - wins) - gamma_dn * wins * hog


# ==========================================
# cerebrum/workspace.py
# ==========================================


class Workspace:
    """k<<n latent slots. STRICT one-hot write (slot j content = the single winning module's read);
    soft-weighted aggregation is FORBIDDEN (BAN-1). Broadcast returns slot contents to all modules
    as a top-down prediction (efference copy)."""
    def __init__(self, k_slots, content_dim, device='cpu', dtype=torch.float64):
        self.k = k_slots; self.dim = content_dim
        self.device = device
        self.dtype = dtype
        self.slots = torch.zeros((k_slots, content_dim), device=device, dtype=dtype)

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.slots = safe_to(self.slots, device, self.dtype)
        return self

    def write(self, z, reads):
        assert_one_hot(z, axis=0)                       # raises on soft weights
        z_t = to_tensor(z, self.device, self.dtype)
        reads_t = to_tensor(reads, self.device, self.dtype)
        
        for j in range(self.k):
            winners = torch.nonzero(z_t[:, j] > 0.5, as_tuple=True)[0]
            if winners.numel():
                # reads_t could be a list/tuple of tensors, or a 2D tensor
                self.slots[j] = reads_t[winners[0]]        # strict one-hot read of the winner

    def broadcast(self):
        return self.slots.sum(dim=0)


# ==========================================
# cerebrum/neuromod.py
# ==========================================


class Neuromodulator:
    def __init__(self, cfg, b_T=0.5, a_Pi=2.0, eta0=1.0, device='cpu', dtype=torch.float64):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.r_bar = 0.0
        self.b_T, self.a_Pi, self.eta0 = b_T, a_Pi, eta0

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    def update(self, reward):
        if isinstance(reward, torch.Tensor):
            reward_val = float(reward.item())
        else:
            reward_val = float(reward)
        M = reward_val - self.r_bar
        self.r_bar += (1.0/self.cfg.tau_r) * (reward_val - self.r_bar)  # EMA
        return torch.tensor(M, device=self.device, dtype=self.dtype)

    def temperature(self, M):
        if isinstance(M, torch.Tensor):
            return self.cfg.T_floor + self.b_T * torch.clamp(M, min=0.0)
        return self.cfg.T_floor + self.b_T * max(0.0, float(M))

    def pi_gain(self, M):
        if isinstance(M, torch.Tensor):
            return 1.0 / (1.0 + torch.exp(-self.a_Pi * M))
        return 1.0 / (1.0 + np.exp(-self.a_Pi * float(M)))

    def eta(self, M):
        if isinstance(M, torch.Tensor):
            return self.eta0 * torch.clamp(M, min=0.0)
        return self.eta0 * max(0.0, float(M))

    def t_gate(self, M, eps=1e-3):
        if isinstance(M, torch.Tensor):
            return 1.0 / (torch.abs(M) + eps)
        return 1.0 / (abs(float(M)) + eps)


# ==========================================
# cerebrum/metaplasticity.py
# ==========================================


class MetaplasticFuse:
    """Per-synapse surprise-gated plasticity permission. Reuses the SAME Pi, eps, eligibility that
    drive inference (NO Fisher pass, NO task-boundary, NO stored anchor weights). Low surprise builds
    a consolidation reserve c -> theta->0 (frozen, protects prior tasks); high surprise erodes c ->
    theta->1 (labile, learn-on-surprise). theta multiplies the four-factor weight update."""
    def __init__(self, shape, cfg, device='cpu', dtype=torch.float64):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.c = torch.zeros(shape, device=device, dtype=dtype)            # consolidation reserve in [0, c_max]
        self.S_bar = torch.zeros(shape, device=device, dtype=dtype)        # per-synapse surprise baseline (EMA)

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.c = safe_to(self.c, device, self.dtype)
        self.S_bar = safe_to(self.S_bar, device, self.dtype)
        return self

    def _raw_surprise(self, Pi_post, eps_post, elig):
        # S_raw_ij = |Pi_i * eps_i * e_j|  (precision-weighted error-eligibility magnitude; local)
        Pi_post_t = to_tensor(Pi_post, self.device, self.dtype)
        eps_post_t = to_tensor(eps_post, self.device, self.dtype)
        elig_t = to_tensor(elig, self.device, self.dtype)
        return torch.abs((Pi_post_t * eps_post_t)[:, None] * elig_t[None, :])

    def update(self, Pi_post, eps_post, elig):
        S_raw = self._raw_surprise(Pi_post, eps_post, elig)
        S = S_raw - self.S_bar                       # surprise relative to the (pre-update) baseline
        predictive = (S_raw <= self.S_bar).to(self.dtype)            # [S]_- regime indicator: build c
        surprising = torch.clamp(S, min=0.0)                             # [S]_+ magnitude: erode c
        
        with torch.no_grad():
            dc = self.cfg.alpha_c*predictive*(self.cfg.c_max - self.c) - self.cfg.beta_c*surprising*self.c
            self.c = torch.clamp(self.c + (1.0/self.cfg.tau_c)*dc, 0.0, self.cfg.c_max)
            self.S_bar += (1.0/self.cfg.tau_S) * (S_raw - self.S_bar)   # baseline EMA (after it is used)
            exponent = -self.cfg.g_theta * (S - self.c)
            exponent = torch.clamp(exponent, min=-50.0, max=50.0)
            theta = 1.0 / (1.0 + torch.exp(exponent))  # sigma(g(S - c))
        return theta


# ==========================================
# cerebrum/plasticity.py
# ==========================================


class Eligibility:
    """Synapse-local presynaptic low-pass trace: tau_e de/dt = -e + a_pre (bare, no Pi inside)."""
    def __init__(self, shape, cfg, device='cpu', dtype=torch.float64):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.value = torch.zeros(shape, device=device, dtype=dtype)

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.value = safe_to(self.value, device, self.dtype)
        return self

    def step(self, a_pre):
        a_pre_t = to_tensor(a_pre, self.device, self.dtype)
        with torch.no_grad():
            self.value += (1.0/self.cfg.tau_e)*(a_pre_t - self.value)
        return self.value

def weight_update(M, theta, Pi_post, eps_post, elig, eta):
    """Four-factor local Hebbian = eta * M * theta * (Pi_post*eps_post) outer elig.
    Equals eta * (-dF/dW) when M=theta=1 (precision-once convention). theta is (out,in)."""
    device = Pi_post.device if hasattr(Pi_post, 'device') else 'cpu'
    dtype = Pi_post.dtype if hasattr(Pi_post, 'dtype') else torch.float64
    
    M_t = to_tensor(M, device, dtype)
    theta_t = to_tensor(theta, device, dtype)
    Pi_post_t = to_tensor(Pi_post, device, dtype)
    eps_post_t = to_tensor(eps_post, device, dtype)
    elig_t = to_tensor(elig, device, dtype)
    
    post = (Pi_post_t * eps_post_t)[:, None]          # (out,1)
    pre = elig_t[None, :]                           # (1,in)
    with torch.no_grad():
        val = eta * M_t * theta_t * (post @ pre)
    return val

def precision_update(Pi, eps_sq, cfg):
    """Diagonal, local-per-unit precision learning. Relaxes Pi toward its spec fixed point
    Pi -> 1/(sigma0^2 + <eps^2>) (spec eq. precision learning). kappa_pi sets the relaxation
    gain, tau_pi the timescale; sigma0 is the precision-floor variance. Local: each unit i
    only reads its own eps_i^2 and Pi_i, no cross-unit / matrix-inverse term."""
    device = Pi.device if hasattr(Pi, 'device') else 'cpu'
    dtype = Pi.dtype if hasattr(Pi, 'dtype') else torch.float64
    
    Pi_t = to_tensor(Pi, device, dtype)
    eps_sq_t = to_tensor(eps_sq, device, dtype)
    target = 1.0 / torch.clamp(cfg.sigma0**2 + eps_sq_t, min=1e-6)
    with torch.no_grad():
        dPi = cfg.kappa_pi * (target - Pi_t)
        val = Pi_t + (1.0/cfg.tau_pi)*dPi
    return val

def feedback_update(B, a_up, eps, cfg):
    """Local feedback-weight rule: eta_b * a_up outer eps - lam_b * B. No transpose of W is read."""
    device = B.device if hasattr(B, 'device') else 'cpu'
    dtype = B.dtype if hasattr(B, 'dtype') else torch.float64
    
    B_t = to_tensor(B, device, dtype)
    a_up_t = to_tensor(a_up, device, dtype)
    eps_t = to_tensor(eps, device, dtype)
    with torch.no_grad():
        val = cfg.eta_b * torch.outer(a_up_t, eps_t) - cfg.lam_b * B_t
    return val

def feedback_update_kp(B, M, Pi_post, eps_post, elig, eta, lam_kp):
    """Kolen-Pollack feedback-alignment rule (OPT-IN). Drives B[l] (shape (out_up, in_post.T)
    i.e. (d[l+1], d[l])) toward W[l].T using the SAME local four-factor product that updates
    W[l] -- only TRANSPOSED -- plus a MATCHED symmetric weight decay.
    """
    device = B.device if hasattr(B, 'device') else 'cpu'
    dtype = B.dtype if hasattr(B, 'dtype') else torch.float64
    
    B_t = to_tensor(B, device, dtype)
    M_t = to_tensor(M, device, dtype)
    Pi_post_t = to_tensor(Pi_post, device, dtype)
    eps_post_t = to_tensor(eps_post, device, dtype)
    elig_t = to_tensor(elig, device, dtype)
    
    post = Pi_post_t * eps_post_t
    pre = elig_t
    with torch.no_grad():
        val = eta * M_t * torch.outer(pre, post) - lam_kp * B_t
    return val


# ==========================================
# cerebrum/unified.py
# ==========================================

"""I5-Unified — ONE coherent network exercising all FIVE CEREBRUM pillars together.

`CerebrumNet` fuses the three staged prototypes into a single `step(obs_slices, action, reward)`:

  Stage-1  Predictive coding + structured grid prior
      - a shared grid HEAD path-integrates on the EXOGENOUS `action` and produces a top-down
        STRUCTURAL prediction (frozen decode of the completed grid code); each module is a
        hierarchical `PCAreas` (separate error neurons, separate feedback `B`, diagonal `Pi`).
  Stage-2  Basal-ganglia gate + k<<n workspace + thalamo-cortical broadcast
      - every module settles (Langevin noise, T>=T_floor) under BOTH the grid top-down AND the
        previous step's workspace broadcast (efference copy);
      - each module emits a SCALAR own-error bid; a stochastic striatal Go/NoGo competition
        selects a STRICT one-hot winner per slot; the winner's content is written one-hot and
        broadcast back next step. Routing EMERGES — there is no attention/query-key term.
  Stage-3  Surprise-gated metaplastic fuse on the module weights
      - a per-synapse `MetaplasticFuse` (reusing the SAME Pi/eps/eligibility — NO Fisher pass,
        NO anchor, NO task-boundary) produces theta in [0,1] that MULTIPLIES the four-factor
        module weight update, so consolidated synapses freeze while surprising ones stay labile.

The ONLY non-local signal crossing the whole network into any weight update is the single
SCALAR neuromodulator `M = r - r_bar`. The workspace broadcast enters ONLY inference as a
prediction, never a weight update (that would be DFA). This class composes the existing
modules with their CURRENT signatures and does NOT duplicate their internal logic.
"""



class CerebrumNet:
    def __init__(self, n_modules, k_slots, slice_dim, cfg, device='cpu', dtype=torch.float64):
        import threading
        self._lock = threading.RLock()
        self.cfg = cfg
        self.M_ = n_modules
        self.k = k_slots
        self.slice_dim = slice_dim
        self.device = device
        self.dtype = dtype
        
        # each module is a PCAreas whose bottom area is its input slice (mirrors CerebrumWorkspaceNet)
        mdims = (slice_dim,) + tuple(cfg.dims[1:]) if len(cfg.dims) > 1 else (slice_dim, slice_dim)
        self.modules = [PCAreas(replace(cfg, dims=mdims, seed=cfg.seed + i), device=device, dtype=dtype) for i in range(n_modules)]
        self.content_dim = mdims[-1]
        
        # Stage-1 structured prior: ONE shared grid head + a frozen decode U into the module top area
        self.grid = GridHead(cfg, device=device, dtype=dtype)
        self.grid.reset()
        self._U = None          # lazily built frozen decode (grid completion -> top-area prediction)
        
        # Stage-2 routing + workspace
        self.gate = BasalGangliaGate(n_modules, k_slots, cfg, seed=cfg.seed, device=device, dtype=dtype)
        self.workspace = Workspace(k_slots, self.content_dim, device=device, dtype=dtype)
        self.nm = Neuromodulator(cfg, device=device, dtype=dtype)
        self.rng = SeededRNG(cfg.seed, device=device, dtype=dtype)
        self.counters = Counters()
        
        # one eligibility trace AND one metaplastic fuse per module per forward layer
        self.elig = [[Eligibility((m.cfg.dims[l + 1],), cfg, device=device, dtype=dtype) for l in range(m.L - 1)] for m in self.modules]
        self.fuse = [[MetaplasticFuse(m.W[l].shape, cfg, device=device, dtype=dtype) for l in range(m.L - 1)] for m in self.modules]
        
        # test/inspection hooks (not load-bearing for the algorithm)
        self._force_theta = None        # if set, pins every module-layer theta to this constant
        self.last_theta = None
        self.last_top_pred = torch.zeros(self.content_dim, device=device, dtype=dtype)
        self._backend = "numpy"

    def set_backend(self, backend, device="cpu"):
        self._backend = backend
        self.device = device
        if backend == "torch":
            dtype = torch.float32 if (device == "mps" or (isinstance(device, str) and "mps" in device)) else torch.float64
            self.to(device, dtype=dtype)
        return self

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        for mod in self.modules:
            mod.to(device, self.dtype)
        self.grid.to(device, self.dtype)
        if self._U is not None:
            self._U = safe_to(self._U, device, self.dtype)
        self.gate.to(device, self.dtype)
        self.workspace.to(device, self.dtype)
        self.nm.to(device, self.dtype)
        self.rng.to(device, self.dtype)
        for m_elig in self.elig:
            for e in m_elig:
                e.to(device, self.dtype)
        for m_fuse in self.fuse:
            for f in m_fuse:
                f.to(device, self.dtype)
        if isinstance(self.last_top_pred, torch.Tensor):
            self.last_top_pred = safe_to(self.last_top_pred, device, self.dtype)
        if self.last_theta is not None:
            for m_i in range(len(self.last_theta)):
                for layer_idx in range(len(self.last_theta[m_i])):
                    if self.last_theta[m_i][layer_idx] is not None:
                        self.last_theta[m_i][layer_idx] = safe_to(self.last_theta[m_i][layer_idx], device, self.dtype)
        return self

    # ------------------------------------------------------------------ grid prior
    def _top_pred_from_grid(self, obs_dim):
        """Structural top-down prediction: frozen decode of the (path-integrated) grid completion,
        projected into the module top-area dimension. Same pattern as CerebrumCore."""
        rec = self.grid.complete() if self.grid.store is not None else torch.zeros(obs_dim, device=self.device, dtype=self.dtype)
        if self._U is None:
            # Maintain seed parity using NumPy RNG
            rng = np.random.default_rng(self.cfg.seed + 7)
            U_np = 0.1 * rng.standard_normal((self.content_dim, obs_dim))
            self._U = torch.tensor(U_np, device=self.device, dtype=self.dtype)
        self.counters.record_global_infer_vectors(k=1, width=self.content_dim)  # broadcast to top area
        return self._U @ rec

    def _broadcast_for_module(self, mod, wksp):
        """Build the per-area efference-copy structure PCAreas.settle_step expects (broadcast[l]).
        The workspace broadcast enters ONLY the bottom area as a prediction term; other areas get 0.
        This NEVER feeds any weight update (it is removed before the four-factor rule runs)."""
        b = [torch.zeros(mod.cfg.dims[l], device=self.device, dtype=self.dtype) for l in range(mod.L)]
        d0 = mod.cfg.dims[0]
        wksp_t = to_tensor(wksp, self.device, self.dtype)
        p0 = torch.zeros(d0, device=self.device, dtype=self.dtype)
        n = min(d0, wksp_t.numel())
        p0[:n] = wksp_t[:n]
        b[0] = p0
        return b

    # ------------------------------------------------------------------ inference only (no learning)
    def _settle_all(self, obs_slices, top_pred, wksp, T, learn=False):
        """Settle every module under BOTH the grid top-down and the workspace broadcast; return
        scalar own-error energy per module and the module read-out (top-area activity).

        When `learn=True` the presynaptic eligibility trace is advanced INSIDE the settle loop
        (as in the proven continual harness) so the four-factor outer product is pattern-specific
        — eligibility tracks the latent WHILE it settles to this observation. The inference-only
        path leaves eligibility untouched."""
        err_sq = torch.zeros(self.M_, device=self.device, dtype=self.dtype)
        reads = torch.zeros((self.M_, self.content_dim), device=self.device, dtype=self.dtype)
        for m_i, mod in enumerate(self.modules):
            bcast = self._broadcast_for_module(mod, wksp)
            for _ in range(self.cfg.n_settle):
                mod.settle_step(self.rng, T=T, clamp_bottom=obs_slices[m_i],
                                top_pred=top_pred, broadcast=bcast, counters=self.counters)
                if learn:
                    for l in range(mod.L - 1):
                        self.elig[m_i][l].step(a_pre=mod.x[l + 1])
            mod.compute_errors(top_pred=top_pred, broadcast=bcast)
            err_sq[m_i] = sum(torch.sum(e ** 2) for e in mod.eps)
            reads[m_i] = mod.x[-1].clone()
        return err_sq, reads

    def settle_only(self, obs_slices, action: Exogenous, T=None):
        """Run grid path-integration + module settling WITHOUT any plasticity (for measurement).

        `T=None` uses the running inference temperature (>= T_floor, the learning-time regularizer);
        pass `T=0.0` for a deterministic noise-free readout that reflects the learned WEIGHTS rather
        than the settling floor (the same convention the Stage-3 measurement uses)."""
        with self._lock:
            self.grid.transition(action)
            top_pred = self._top_pred_from_grid(self.modules[0].cfg.dims[0])
            self.last_top_pred = top_pred.clone()
            wksp = self.workspace.broadcast()
            T = self.nm.temperature(0.0) if T is None else T
            return self._settle_all(obs_slices, top_pred, wksp, T)

    # ------------------------------------------------------------------ full step
    def step(self, obs_slices, action: Exogenous, reward):
        with self._lock:
            # (0) Sanitize inputs
            # Sanitize reward
            if isinstance(reward, torch.Tensor):
                if not torch.isfinite(reward).all():
                    reward = torch.where(torch.isfinite(reward), reward, torch.zeros_like(reward))
            elif isinstance(reward, np.ndarray):
                if not np.isfinite(reward).all():
                    reward = np.where(np.isfinite(reward), reward, 0.0)
            else:
                try:
                    reward_val = float(reward)
                    if np.isnan(reward_val) or np.isinf(reward_val):
                        reward = 0.0
                    else:
                        reward = reward_val
                except (ValueError, TypeError):
                    reward = 0.0

            # Sanitize action
            if isinstance(action, Exogenous):
                v = action.value
                if isinstance(v, torch.Tensor):
                    if not torch.isfinite(v).all():
                        v = torch.where(torch.isfinite(v), v, torch.zeros_like(v))
                        action = Exogenous(v)
                elif isinstance(v, np.ndarray):
                    if not np.isfinite(v).all():
                        v = np.where(np.isfinite(v), v, 0.0)
                        action = Exogenous(v)
                else:
                    v_arr = np.asarray(v)
                    if not np.isfinite(v_arr).all():
                        v_arr = np.where(np.isfinite(v_arr), v_arr, 0.0)
                        action = Exogenous(v_arr)

            # Sanitize obs_slices
            sanitized_obs_slices = []
            for obs in obs_slices:
                if isinstance(obs, (list, tuple)):
                    for item in obs:
                        if isinstance(item, (str, dict)):
                            raise TypeError("Observations must be numeric.")
                try:
                    if isinstance(obs, torch.Tensor):
                        if obs.dtype not in (torch.float16, torch.float32, torch.float64, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                            raise TypeError("Observations must be numeric.")
                        obs_conv = obs.to(device=self.device, dtype=self.dtype)
                    elif isinstance(obs, np.ndarray):
                        if obs.dtype.kind not in 'bifc':
                            raise TypeError("Observations must be numeric.")
                        obs_conv = torch.as_tensor(obs, device=self.device, dtype=self.dtype)
                    else:
                        arr = np.array(obs, dtype=np.float64)
                        if arr.dtype.kind not in 'bifc':
                            raise TypeError("Observations must be numeric.")
                        obs_conv = torch.as_tensor(arr, device=self.device, dtype=self.dtype)
                except (ValueError, TypeError) as e:
                    raise TypeError("Observations must be numeric.") from e

                if len(obs_conv) != self.slice_dim:
                    raise ValueError(f"Observation slice length must match slice_dim={self.slice_dim}")

                if isinstance(obs, torch.Tensor):
                    if not torch.isfinite(obs).all():
                        obs = torch.where(torch.isfinite(obs), obs, torch.zeros_like(obs))
                elif isinstance(obs, np.ndarray):
                    if not np.isfinite(obs).all():
                        obs = np.where(np.isfinite(obs), obs, 0.0)
                else:
                    obs = obs_conv
                sanitized_obs_slices.append(obs)
            obs_slices = sanitized_obs_slices

            # (1) Stage-1: grid HEAD path-integrates on the EXOGENOUS action (BAN-5 enforced inside
            #     transition via assert_exogenous_action) -> structural top-down prediction.
            self.grid.transition(action)
            top_pred = self._top_pred_from_grid(self.modules[0].cfg.dims[0])
            self.last_top_pred = top_pred.clone()
            
            # Bind the (mean) observation into the grid content store, GATED by the scalar reward-
            # prediction-error preview M = r - r_bar (read BEFORE the EMA update below consumes it).
            M_preview = float(reward) - self.nm.r_bar
            obs_tensors = [to_tensor(o, self.device, self.dtype) for o in obs_slices]
            obs_mean = torch.stack(obs_tensors, dim=0).mean(dim=0)
            self.grid.bind(obs_mean, M=max(M_preview, 0.0))

            # (2) Stage-2: settle every module under grid top-down + previous workspace broadcast.
            wksp = self.workspace.broadcast()
            self.counters.record_global_infer_vectors(k=self.k, width=self.content_dim)
            T = self.nm.temperature(0.0)
            err_sq, reads = self._settle_all(obs_slices, top_pred, wksp, T, learn=True)

            # (3) Stage-2 routing: scalar own-error bid -> stochastic one-hot select -> one-hot write.
            pi = torch.tensor([float(torch.mean(mod.Pi[-1]).item()) for mod in self.modules], device=self.device, dtype=self.dtype)
            bids = self.gate.bid(err_sq=err_sq, pi=pi)
            T_gate = self.cfg.gate_temp if self.cfg.gate_temp > 0.0 else self.nm.t_gate(max(reward, 1e-3))
            z = self.gate.select(bids, self.rng, T_gate=T_gate)
            self.workspace.write(z, reads)        # asserts one-hot inside (BAN-1)

            # (4) Learn: single SCALAR M gates everything; per-synapse metaplastic theta gates the
            #     four-factor module weight update; gate learning + reward-aware homeostasis.
            M = self.nm.update(reward)
            assert_scalar_M(M)                    # BAN-2
            self.counters.record_global_learn(1)  # O(1) scalar-M learn-time global comm
            self.last_theta = [[None] * (mod.L - 1) for mod in self.modules]
            
            with torch.no_grad():
                for m_i, mod in enumerate(self.modules):
                    for l in range(mod.L - 1):
                        theta = self.fuse[m_i][l].update(mod.Pi[l], mod.eps[l], self.elig[m_i][l].value)
                        if self._force_theta is not None:
                            theta = torch.full_like(mod.W[l], float(self._force_theta), device=self.device, dtype=self.dtype)
                        self.last_theta[m_i][l] = theta
                        
                        dW = weight_update(M=M, theta=theta, Pi_post=mod.Pi[l],
                                                  eps_post=mod.eps[l], elig=self.elig[m_i][l].value,
                                                  eta=self.cfg.eta_w / self.cfg.tau_w)
                        if self.cfg.align_feedback:
                            mod.W[l] += dW - self.cfg.lam_kp * mod.W[l]
                            mod.B[l] += feedback_update_kp(mod.B[l], M=M, Pi_post=mod.Pi[l],
                                                           eps_post=mod.eps[l], elig=self.elig[m_i][l].value,
                                                           eta=self.cfg.eta_w / self.cfg.tau_w,
                                                           lam_kp=self.cfg.lam_kp)
                        else:
                            mod.W[l] += dW
                            dB = (1.0 / self.cfg.tau_b) * feedback_update(mod.B[l], a_up=mod.x[l + 1],
                                                                                 eps=mod.eps[l], cfg=self.cfg)
                            mod.B[l] += dB
                        
                        mod.Pi[l] = precision_update(mod.Pi[l], eps_sq=mod.eps[l] ** 2, cfg=self.cfg)
                self.gate.learn(M=M)
                self.gate.homeostasis(M=M)            # reward-aware homeostasis (spec FM5b)
                
            return z, M


# ==========================================
# cerebrum/workspace_net.py
# ==========================================



class CerebrumWorkspaceNet:
    """Stage-2 cortical workspace network: M modules compete via a stochastic gate for k slots;
    winners' content is broadcast back as top-down prediction. Routing EMERGES from the loop;
    there is no attention/mixer module."""
    def __init__(self, n_modules, k_slots, slice_dim, cfg, device='cpu', dtype=torch.float64):
        self.cfg = cfg
        self.M_ = n_modules
        self.k = k_slots
        self.device = device
        self.dtype = dtype
        
        # each module is a PCAreas whose bottom area = its input slice
        mdims = (slice_dim,) + tuple(cfg.dims[1:]) if len(cfg.dims) > 1 else (slice_dim, slice_dim)
        self.modules = [PCAreas(replace(cfg, dims=mdims, seed=cfg.seed+i), device=device, dtype=dtype) for i in range(n_modules)]
        self.content_dim = mdims[-1]
        
        self.gate = BasalGangliaGate(n_modules, k_slots, cfg, seed=cfg.seed, device=device, dtype=dtype)
        self.workspace = Workspace(k_slots, self.content_dim, device=device, dtype=dtype)
        self.nm = Neuromodulator(cfg, device=device, dtype=dtype)
        self.rng = SeededRNG(cfg.seed, device=device, dtype=dtype)
        self.counters = Counters()
        
        self.elig = [[Eligibility((m.cfg.dims[l+1],), cfg, device=device, dtype=dtype) for l in range(m.L-1)] for m in self.modules]

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        for mod in self.modules:
            mod.to(device, self.dtype)
        self.gate.to(device, self.dtype)
        self.workspace.to(device, self.dtype)
        self.nm.to(device, self.dtype)
        self.rng.to(device, self.dtype)
        for m_elig in self.elig:
            for e in m_elig:
                e.to(device, self.dtype)
        return self

    def step(self, obs_slices, reward):
        bcast = self.workspace.broadcast()                          # top-down efference copy from last step
        self.counters.record_global_infer_vectors(k=self.k, width=self.content_dim)
        
        bcast_t = to_tensor(bcast, self.device, self.dtype)
        # top-down prediction to each module's top area = a frozen projection of the broadcast
        if bcast_t.numel() >= self.content_dim:
            top_pred = bcast_t[:self.content_dim]
        else:
            top_pred = torch.zeros(self.content_dim, device=self.device, dtype=self.dtype)
            
        # 1) settle every module with the broadcast as top-down
        T = self.nm.temperature(0.0)
        err_sq = torch.zeros(self.M_, device=self.device, dtype=self.dtype)
        reads = torch.zeros((self.M_, self.content_dim), device=self.device, dtype=self.dtype)
        for m_i, mod in enumerate(self.modules):
            for _ in range(self.cfg.n_settle):
                mod.settle_step(self.rng, T=T, clamp_bottom=obs_slices[m_i],
                                top_pred=top_pred, counters=self.counters)
            mod.compute_errors(top_pred=top_pred)
            err_sq[m_i] = sum(torch.sum(e**2) for e in mod.eps)
            reads[m_i] = mod.x[-1].clone()                            # module content = top-area activity
            
        # 2) gate: bid (scalar own-error) -> stochastic one-hot select -> write -> broadcast
        pi = torch.tensor([float(torch.mean(mod.Pi[-1]).item()) for mod in self.modules], device=self.device, dtype=self.dtype)
        bids = self.gate.bid(err_sq=err_sq, pi=pi)
        T_gate = self.cfg.gate_temp if self.cfg.gate_temp > 0.0 else self.nm.t_gate(max(reward, 1e-3))
        z = self.gate.select(bids, self.rng, T_gate=T_gate)
        self.workspace.write(z, reads)
        
        # 3) learn: scalar M gates module plasticity + gate learning + homeostasis
        M = self.nm.update(reward)
        assert_scalar_M(M)
        self.counters.record_global_learn(1)
        
        with torch.no_grad():
            for m_i, mod in enumerate(self.modules):
                for l in range(mod.L-1):
                    self.elig[m_i][l].step(a_pre=mod.x[l+1])
                    eta_w = self.cfg.eta_w/self.cfg.tau_w
                    dW = weight_update(M=M, theta=torch.ones_like(mod.W[l]), Pi_post=mod.Pi[l],
                                       eps_post=mod.eps[l], elig=self.elig[m_i][l].value, eta=eta_w)
                    if self.cfg.align_feedback:
                        mod.W[l] += dW - self.cfg.lam_kp*mod.W[l]
                        mod.B[l] += feedback_update_kp(mod.B[l], M=M, Pi_post=mod.Pi[l],
                                       eps_post=mod.eps[l], elig=self.elig[m_i][l].value,
                                       eta=eta_w, lam_kp=self.cfg.lam_kp)
                    else:
                        mod.W[l] += dW
                        mod.B[l] += (1.0/self.cfg.tau_b)*feedback_update(mod.B[l], a_up=mod.x[l+1], eps=mod.eps[l], cfg=self.cfg)
                    mod.Pi[l] = precision_update(mod.Pi[l], eps_sq=mod.eps[l]**2, cfg=self.cfg)
            self.gate.learn(M=M)
            self.gate.homeostasis(M=M)   # reward-aware homeostasis (spec FM5b)
            
        return z, M


# ==========================================
# cerebrum/core_net.py
# ==========================================


class CerebrumCore:
    def __init__(self, cfg, device='cpu', dtype=torch.float64):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        
        self.pc = PCAreas(cfg, device=device, dtype=dtype)
        self.grid = GridHead(cfg, device=device, dtype=dtype)
        self.grid.reset()
        self.nm = Neuromodulator(cfg, device=device, dtype=dtype)
        self.rng = SeededRNG(cfg.seed, device=device, dtype=dtype)
        self.counters = Counters()
        self._U = None
        self.elig = [Eligibility((cfg.dims[l+1],), cfg, device=device, dtype=dtype) for l in range(self.pc.L-1)]

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.pc.to(device, self.dtype)
        self.grid.to(device, self.dtype)
        self.nm.to(device, self.dtype)
        self.rng.to(device, self.dtype)
        if self._U is not None:
            self._U = safe_to(self._U, device, self.dtype)
        for e in self.elig:
            e.to(device, self.dtype)
        return self

    def _top_pred_from_grid(self, obs_dim):
        rec = self.grid.complete() if self.grid.store is not None else torch.zeros(obs_dim, device=self.device, dtype=self.dtype)
        if self._U is None:
            rng = np.random.default_rng(self.cfg.seed+7)
            U_np = 0.1*rng.standard_normal((self.cfg.dims[-1], obs_dim))
            self._U = torch.tensor(U_np, device=self.device, dtype=self.dtype)
        self.counters.record_global_infer_vectors(k=1, width=self.cfg.dims[-1])
        return self._U @ rec

    def move(self, action: Exogenous):
        self.grid.transition(action)

    def observe_and_learn(self, obs, reward):
        obs_t = to_tensor(obs, self.device, self.dtype)
        self.grid.bind(obs_t, M=1.0)
        top_pred = self._top_pred_from_grid(obs_t.numel())
        
        T = self.nm.temperature(0.0)
        for _ in range(self.cfg.n_settle):
            self.pc.settle_step(self.rng, T=T, clamp_bottom=obs_t, top_pred=top_pred,
                                counters=self.counters)
        self.pc.compute_errors(top_pred=top_pred)
        
        M = self.nm.update(reward)
        assert_scalar_M(M)
        self.counters.record_global_learn(1)
        
        with torch.no_grad():
            for l in range(self.pc.L-1):
                self.elig[l].step(a_pre=self.pc.x[l+1])
                eta_w = self.cfg.eta_w/self.cfg.tau_w
                dW = weight_update(M=M, theta=torch.ones_like(self.pc.W[l]),
                                   Pi_post=self.pc.Pi[l], eps_post=self.pc.eps[l],
                                   elig=self.elig[l].value, eta=eta_w)
                if self.cfg.align_feedback:
                    self.pc.W[l] += dW - self.cfg.lam_kp*self.pc.W[l]
                    self.pc.B[l] += feedback_update_kp(self.pc.B[l], M=M, Pi_post=self.pc.Pi[l],
                                       eps_post=self.pc.eps[l], elig=self.elig[l].value,
                                       eta=eta_w, lam_kp=self.cfg.lam_kp)
                else:
                    self.pc.W[l] += dW
                    self.pc.B[l] += (1.0/self.cfg.tau_b)*feedback_update(self.pc.B[l],
                                       a_up=self.pc.x[l+1], eps=self.pc.eps[l], cfg=self.cfg)
                self.pc.Pi[l] = precision_update(self.pc.Pi[l], eps_sq=self.pc.eps[l]**2, cfg=self.cfg)
        return M

    def predict_obs_here(self, obs_dim):
        return self.grid.complete() if self.grid.store is not None else torch.zeros(obs_dim, device=self.device, dtype=self.dtype)


# ==========================================
# cerebrum/energy.py
# ==========================================

"""Energy / operation accounting for CEREBRUM — success-axis-2 instrumentation.

The honest energy story (spec section neuromorphic_mapping): only DYNAMIC switching energy decays
with competence (as prediction error eps -> 0, error neurons fall silent and stop driving their
synapses). Learn-time global communication is O(1) — a single scalar neuromodulator M — whereas a
matched backprop network broadcasts an error VECTOR per layer (O(depth) elements). Static/leakage
power and settle-time energy do NOT decay; only the event-driven dynamic term does."""


def _to_float(val):
    if hasattr(val, 'item'):
        return float(val.item())
    return float(val)


def _numel(x):
    if hasattr(x, 'numel'):
        return x.numel()
    if hasattr(x, 'shape'):
        return int(np.prod(x.shape))
    return len(x)


def spike_sparsity(eps_list, tol=1e-6):
    """Fraction of error-neuron units that are ACTIVE (|eps| > tol). Event-driven: well-predicted
    units are silent, so this falls toward a floor as the network becomes competent."""
    active = 0
    total = 0
    for e in eps_list:
        if isinstance(e, (torch.Tensor, TensorSliceWrapper)):
            active += int(torch.sum(torch.abs(e) > tol).item())
        else:
            active += int(np.sum(np.abs(e) > tol))
        total += _numel(e)
    return active / total if total else 0.0


def dynamic_synaptic_ops(net, tol=1e-6):
    """Event-driven synaptic-op count: a forward synapse computes only when its postsynaptic error
    neuron spikes. ops = sum_l (#active eps_l) * fan-in. Silent error neurons cost ~0, so the
    dynamic op count decays with competence."""
    ops = 0
    for l in range(net.L - 1):
        eps = net.eps[l]
        if isinstance(eps, (torch.Tensor, TensorSliceWrapper)):
            active = int(torch.sum(torch.abs(eps) > tol).item())
        else:
            active = int(np.sum(np.abs(eps) > tol))
        ops += active * net.W[l].shape[1]      # each active error unit drives its fan-in synapses
    return ops


def dynamic_energy_magnitude(net):
    """Magnitude-weighted dynamic switching-energy proxy: sum over predicted areas of (total error
    activity Σ|eps_l|) * fan-in. In graded event-driven coding the switching energy scales with total
    error activity, so this decays SMOOTHLY as the network becomes competent (eps -> 0). It is the
    robust headline energy metric; the thresholded spike count is a conservative companion."""
    e = 0.0
    for l in range(net.L - 1):
        eps = net.eps[l]
        if isinstance(eps, (torch.Tensor, TensorSliceWrapper)):
            sum_eps = torch.sum(torch.abs(eps)).item()
        else:
            sum_eps = np.sum(np.abs(eps))
        e += float(sum_eps) * net.W[l].shape[1]
    return e


def dense_backprop_ops(dims):
    """Dense forward+backward MAC count for a matched backprop net: every synapse computes every
    step (rho = 1), forward AND backward. The comparator CEREBRUM's event-driven sparsity undercuts."""
    fwd = sum(dims[l] * dims[l + 1] for l in range(len(dims) - 1))
    return 2 * fwd     # forward + backward dense passes


def global_comm_per_update(dims):
    """Global-communication events crossing the whole network per WEIGHT UPDATE.
    CEREBRUM: ONE scalar neuromodulator M (a single diffuse wire). Backprop: an error VECTOR at every
    layer (O(depth) vector elements that must be transported between layers)."""
    return {
        "cerebrum_learn_scalars": 1,
        "backprop_error_vector_elems": int(sum(dims[1:])),   # error vectors at each non-input layer
    }


# ==========================================
# cerebrum/grounding/sensory.py
# ==========================================


class SensoryProcessor:
    """Transforms raw sensor readings into a normalized 5-dimensional workspace state."""
    def process(self, lidar_data, camera_data, odometer_data):
        # 1. Protection against None, NaN, and Inf
        if lidar_data is None:
            lidar_data = np.array([])
        else:
            lidar_data = np.asarray(lidar_data, dtype=float)
            lidar_data = np.where(np.isnan(lidar_data) | np.isinf(lidar_data), 10.0, lidar_data)
            
        if camera_data is None:
            camera_data = np.array([])
        else:
            camera_data = np.asarray(camera_data, dtype=float)
            camera_data = np.where(np.isnan(camera_data) | np.isinf(camera_data), 0.0, camera_data)
            
        if odometer_data is None:
            odometer_data = np.array([])
        else:
            odometer_data = np.asarray(odometer_data, dtype=float)
            odometer_data = np.where(np.isnan(odometer_data) | np.isinf(odometer_data), 0.0, odometer_data)
        
        # 2. Extract min lidar
        min_lidar = float(np.min(lidar_data)) if len(lidar_data) > 0 else 1.0
        
        # 3. Visual splits
        left_slice = camera_data[:len(camera_data)//2] if len(camera_data) > 0 else np.array([])
        right_slice = camera_data[len(camera_data)//2:] if len(camera_data) > 0 else np.array([])
        left_cam = float(np.mean(left_slice)) if len(left_slice) > 0 else 0.0
        right_cam = float(np.mean(right_slice)) if len(right_slice) > 0 else 0.0
        
        # 4. Odometry
        velocity = float(odometer_data[0]) if len(odometer_data) > 0 else 0.0
        heading = float(odometer_data[1]) if len(odometer_data) > 1 else 0.0
        
        # 5. Clamping
        min_lidar = np.clip(min_lidar, 0.0, 10.0)
        left_cam = np.clip(left_cam, 0.0, 1.0)
        right_cam = np.clip(right_cam, 0.0, 1.0)
        
        return np.array([min_lidar, left_cam, right_cam, velocity, heading], dtype=float)


# ==========================================
# cerebrum/grounding/motor.py
# ==========================================


class MotorProcessor:
    """Maps continuous workspace actions or routing indexes to wheel velocities."""
    def __init__(self, mode="discrete", W_motor=None, b_motor=None, u_sat=2.0):
        self.mode = mode
        self.W_motor = W_motor
        self.b_motor = b_motor
        self.u_sat = u_sat
        
    def process(self, action_vector):
        if action_vector is None:
            action_vector = np.array([])
        else:
            action_vector = np.asarray(action_vector, dtype=float)
            action_vector = np.where(np.isnan(action_vector) | np.isinf(action_vector), 0.0, action_vector)
            
        if len(action_vector) == 0:
            return np.array([0.0, 0.0])
            
        if self.mode == "linear" and self.W_motor is not None:
            try:
                # Linear readout mapping
                W_motor = np.asarray(self.W_motor)
                if W_motor.ndim >= 3:
                    W_motor = W_motor.reshape(W_motor.shape[0], -1)
                
                if W_motor.ndim == 0:
                    vels = np.zeros(2)
                elif W_motor.ndim == 1:
                    expected_dim = W_motor.shape[0]
                    if len(action_vector) != expected_dim:
                        vels = np.zeros(2)
                    else:
                        res = np.dot(W_motor, action_vector)
                        vels = np.array([res, 0.0]) if np.isscalar(res) else np.asarray(res)
                else:
                    expected_dim = W_motor.shape[1]
                    if len(action_vector) != expected_dim:
                        output_dim = W_motor.shape[0]
                        vels = np.zeros(output_dim)
                    else:
                        b_val = self.b_motor if self.b_motor is not None else np.zeros(W_motor.shape[0])
                        b_val = np.asarray(b_val, dtype=float)
                        if b_val.ndim >= 2:
                            b_val = b_val.flatten()
                        if b_val.shape[0] != W_motor.shape[0]:
                            b_val = np.zeros(W_motor.shape[0])
                        vels = np.dot(W_motor, action_vector) + b_val
            except (ValueError, TypeError, AttributeError):
                vels = np.zeros(2)
        else:
            # Discrete workspace gating mapping (Default Mock)
            if np.all(action_vector == 0.0):
                vels = np.array([0.0, 0.0])  # Standby
            else:
                act_idx = np.argmax(action_vector)
                if act_idx == 0:
                    vels = np.array([1.0, 1.0])  # Forward
                elif act_idx == 1:
                    vels = np.array([-0.5, 0.5])  # Left turn
                elif act_idx == 2:
                    vels = np.array([0.5, -0.5])  # Right turn
                else:
                    vels = np.array([0.0, 0.0])  # Standby
                
        vels = np.asarray(vels)
        if np.isnan(vels).any() or np.isinf(vels).any():
            vels = np.zeros_like(vels)
        return np.clip(vels, -self.u_sat, self.u_sat)


# ==========================================
# cerebrum/grounding/physics.py
# ==========================================

import sys

# Try to import real pybullet
try:
    import pybullet as real_p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False

class MockPyBullet:
    GUI = 1
    DIRECT = 2
    
    def __init__(self):
        self.connected = False
        self.gravity = [0.0, 0.0, -9.81]
        self.bodies = {}
        self.time_step = 1.0 / 240.0
        
    def connect(self, connection_mode=1):
        self.connected = True
        if PYBULLET_AVAILABLE:
            try:
                return real_p.connect(connection_mode)
            except Exception:
                pass
        return 0
        
    def disconnect(self):
        self.connected = False
        if PYBULLET_AVAILABLE:
            try:
                real_p.disconnect()
            except Exception:
                pass
        
    def setGravity(self, x, y, z):
        self.gravity = [x, y, z]
        if PYBULLET_AVAILABLE:
            try:
                real_p.setGravity(x, y, z)
            except Exception:
                pass
        
    def loadURDF(self, urdf_path, basePosition=(0.0, 0.0, 0.0), baseOrientation=(0.0, 0.0, 0.0, 1.0)):
        if baseOrientation is not None and len(baseOrientation) != 4:
            raise ValueError("Orientation must be a 4-element quaternion.")
        body_id = len(self.bodies) + 1
        self.bodies[body_id] = {
            "path": urdf_path,
            "pos": np.array(basePosition, dtype=float),
            "orn": np.array(baseOrientation, dtype=float),
            "vel": np.zeros(3, dtype=float),
            "omega": np.zeros(3, dtype=float),
            "joints": {0: 0.0, 1: 0.0}
        }
        if PYBULLET_AVAILABLE:
            try:
                real_id = real_p.loadURDF(urdf_path, basePosition, baseOrientation)
                # Keep tracking in self.bodies just in case
                self.bodies[real_id] = self.bodies.pop(body_id)
                return real_id
            except Exception:
                pass
        return body_id
        
    def resetBasePositionAndOrientation(self, body_id, position, orientation):
        if len(orientation) != 4:
            raise ValueError("Orientation must have exactly 4 elements.")
        if body_id in self.bodies:
            self.bodies[body_id]["pos"] = np.array(position, dtype=float)
            self.bodies[body_id]["orn"] = np.array(orientation, dtype=float)
        if PYBULLET_AVAILABLE:
            try:
                real_p.resetBasePositionAndOrientation(body_id, position, orientation)
            except Exception:
                pass
            
    def getBasePositionAndOrientation(self, body_id):
        if PYBULLET_AVAILABLE:
            try:
                return real_p.getBasePositionAndOrientation(body_id)
            except Exception:
                pass
        if body_id in self.bodies:
            b = self.bodies[body_id]
            return b["pos"].tolist(), b["orn"].tolist()
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
        
    def setJointMotorControl2(self, bodyUniqueId, jointIndex, controlMode, targetVelocity=None, targetPosition=None, force=None, **kwargs):
        if PYBULLET_AVAILABLE:
            try:
                real_p.setJointMotorControl2(
                    bodyUniqueId, jointIndex, controlMode,
                    targetVelocity=targetVelocity, targetPosition=targetPosition,
                    force=force, **kwargs
                )
            except Exception:
                pass
        if bodyUniqueId in self.bodies:
            b = self.bodies[bodyUniqueId]
            if targetVelocity is not None:
                b["joints"][jointIndex] = targetVelocity
            elif targetPosition is not None:
                b["joints"][jointIndex] = targetPosition
            
            l_vel = b["joints"].get(0, 0.0)
            r_vel = b["joints"].get(1, 0.0)
            forward_speed = 0.5 * (l_vel + r_vel)
            yaw_rate = 0.5 * (r_vel - l_vel)
            
            b["vel"][0] = forward_speed
            b["omega"][2] = yaw_rate
            
    def stepSimulation(self):
        stepped_real = False
        if PYBULLET_AVAILABLE:
            try:
                real_p.stepSimulation()
                stepped_real = True
            except Exception:
                pass
        if stepped_real:
            return
        for b in self.bodies.values():
            z = b["orn"][2]
            w = b["orn"][3]
            yaw = 2.0 * np.arctan2(z, w)
            
            yaw += b["omega"][2] * self.time_step
            new_z = np.sin(yaw / 2.0)
            new_w = np.cos(yaw / 2.0)
            b["orn"] = np.array([0.0, 0.0, new_z, new_w], dtype=float)
            
            dx = b["vel"][0] * np.cos(yaw)
            dy = b["vel"][0] * np.sin(yaw)
            b["pos"][0] += dx * self.time_step
            b["pos"][1] += dy * self.time_step
            b["pos"][2] += b["vel"][2] * self.time_step
            
    def getKeyboardEvents(self):
        if PYBULLET_AVAILABLE:
            try:
                return real_p.getKeyboardEvents()
            except Exception:
                pass
        return {}
        
    def getLinkState(self, bodyUniqueId, linkIndex):
        if PYBULLET_AVAILABLE:
            try:
                return real_p.getLinkState(bodyUniqueId, linkIndex)
            except Exception:
                pass
        if bodyUniqueId in self.bodies:
            b = self.bodies[bodyUniqueId]
            return [b["pos"].tolist(), b["orn"].tolist()]
        return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]

# Register pybullet in sys.modules if not present
if "pybullet" not in sys.modules:
    sys.modules["pybullet"] = MockPyBullet()


# ==========================================
# cerebrum/grounding/ros_node.py
# ==========================================

import sys

# We need to import our grounding components

# Try to import real rclpy
try:
    import rclpy
    from rclpy.node import Node as ROSNode
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    
# Import message types
try:
    from std_msgs.msg import Float64MultiArray as ROSFloat64MultiArray
    STD_MSGS_AVAILABLE = True
except ImportError:
    STD_MSGS_AVAILABLE = False

# Fallback definition for Mock classes
class MockPublisher:
    _queue = []
    _processing = False

    def __init__(self, topic_name, msg_type):
        self.topic_name = topic_name
        self.msg_type = msg_type
        self.published_messages = []
        
    def publish(self, msg):
        self.published_messages.append(msg)
        MockPublisher._queue.append((self, msg))
        if not MockPublisher._processing:
            MockPublisher._processing = True
            try:
                while MockPublisher._queue:
                    pub, current_msg = MockPublisher._queue.pop(0)
                    if pub.topic_name in MockRclpy.subscriptions:
                        for sub in MockRclpy.subscriptions[pub.topic_name]:
                            sub.callback(current_msg)
            finally:
                MockPublisher._processing = False

class MockSubscription:
    def __init__(self, topic_name, msg_type, callback):
        self.topic_name = topic_name
        self.msg_type = msg_type
        self.callback = callback

class MockLogger:
    def info(self, msg):  pass
    def warn(self, msg):  pass
    def error(self, msg): pass

class MockNode:
    def __init__(self, node_name):
        self.node_name = node_name
        self.publishers = {}
        self.subscriptions = {}
        self.logger = MockLogger()
        
    def create_publisher(self, msg_type, topic_name, qos_profile=10):
        pub = MockPublisher(topic_name, msg_type)
        self.publishers[topic_name] = pub
        MockRclpy.publishers.setdefault(topic_name, []).append(pub)
        return pub
        
    def create_subscription(self, msg_type, topic_name, callback, qos_profile=10):
        sub = MockSubscription(topic_name, msg_type, callback)
        self.subscriptions[topic_name] = sub
        MockRclpy.subscriptions.setdefault(topic_name, []).append(sub)
        return sub
        
    def get_logger(self):
        return self.logger
        
    def destroy_node(self):
        pass

class MockRclpy:
    publishers = {}
    subscriptions = {}
    initialized = False
    
    @classmethod
    def init(cls, args=None):
        cls.publishers.clear()
        cls.subscriptions.clear()
        cls.initialized = True
        MockPublisher._queue = []
        MockPublisher._processing = False
        
    @classmethod
    def shutdown(cls):
        cls.publishers.clear()
        cls.subscriptions.clear()
        cls.initialized = False
        MockPublisher._queue = []
        MockPublisher._processing = False
        
    @classmethod
    def create_node(cls, node_name):
        if not cls.initialized:
            raise RuntimeError("rclpy not initialized")
        return MockNode(node_name)
        
    @classmethod
    def spin_once(cls, node, timeout_sec=0.0):
        import time
        time.sleep(timeout_sec)

class std_msgs:
    class msg:
        class Float64MultiArray:
            def __init__(self):
                self.data = []

# Register mocks in sys.modules if real ones not present
if not ROS_AVAILABLE:
    sys.modules["rclpy"] = MockRclpy
    RclpyClass = MockRclpy
    NodeClass = MockNode
else:
    sys.modules.setdefault("rclpy", rclpy)
    RpyClass = rclpy
    NodeClass = ROSNode

if not STD_MSGS_AVAILABLE:
    sys.modules["std_msgs"] = std_msgs
    sys.modules["std_msgs.msg"] = std_msgs.msg
    Float64MultiArrayClass = std_msgs.msg.Float64MultiArray
else:
    sys.modules.setdefault("std_msgs", sys.modules.get("std_msgs"))
    Float64MultiArrayClass = ROSFloat64MultiArray

class CerebrumROSNode(NodeClass):
    def __init__(self, net, node_name="cerebrum_ros_node", reflex=None, sensory_processor=None, motor_processor=None, is_direct_state=False):
        super().__init__(node_name)
        import threading
        self._lock = threading.RLock()
        self.net = net
        self.reflex = reflex
        self.sensory_processor = sensory_processor or SensoryProcessor()
        self.motor_processor = motor_processor or MotorProcessor()
        self.is_direct_state = is_direct_state
        self.reward = 1.0  # Default initial reward
        
        # Publishers
        self.motor_pub = self.create_publisher(Float64MultiArrayClass, "/motor_commands")
        self.telemetry_pub = self.create_publisher(Float64MultiArrayClass, "/telemetry")
        
        # Subscriptions
        self.sensory_sub = self.create_subscription(
            Float64MultiArrayClass,
            "/sensory_input",
            self.sensory_callback
        )
        self.reward_sub = self.create_subscription(
            Float64MultiArrayClass,
            "/reward",
            self.reward_callback
        )
        
    def reward_callback(self, msg):
        with self._lock:
            try:
                if msg is None or not hasattr(msg, 'data'):
                    self.get_logger().warn("Malformed reward message: msg has no data attribute.")
                    return
                if len(msg.data) > 0:
                    val = float(msg.data[0])
                    if np.isnan(val) or np.isinf(val):
                        self.get_logger().warn("NaN/Inf received in reward callback, skipping.")
                        return
                    self.reward = val
            except (TypeError, ValueError) as e:
                self.get_logger().error(f"Error processing reward message: {e}")
            
    def sensory_callback(self, msg):
        with self._lock:
            try:
                if msg is None or not hasattr(msg, 'data'):
                    self.get_logger().warn("Malformed sensory message: msg has no data attribute.")
                    return
                
                # Validate and clean NaN/Inf sensory inputs
                cleaned_data = []
                has_invalid = False
                for x in msg.data:
                    val = float(x)
                    if np.isnan(val) or np.isinf(val):
                        cleaned_data.append(0.0)
                        has_invalid = True
                    else:
                        cleaned_data.append(val)
                if has_invalid:
                    self.get_logger().warn("NaN/Inf detected in sensory input; replacing with 0.0.")
                msg.data = cleaned_data

                data_len = len(msg.data)
                M_ = self.net.M_
                slice_dim = self.net.modules[0].cfg.dims[0]
                
                bypass_active = False
                action_u = None
                
                if self.reflex is not None:
                    if self.is_direct_state and data_len == 5:
                        state = np.asarray(msg.data, dtype=float)
                        # Construct a dictionary matching the semantic positional mapping:
                        # index 0: dist, index 1: tilt, index 2: error_energy
                        state_to_evaluate = {
                            "dist": float(state[0]),
                            "tilt": float(state[1]),
                            "error_energy": float(state[2])
                        }
                    else:
                        if data_len >= 8:
                            lidar = msg.data[:4]
                            camera = msg.data[4:6]
                            odometer = msg.data[6:8]
                        else:
                            lidar = msg.data
                            camera = []
                            odometer = []
                        state = self.sensory_processor.process(lidar, camera, odometer)
                        # Construct a dictionary with the correct mapping from processed state:
                        # state[0] = min_lidar (dist)
                        # state[1] = left_cam (camera, NOT tilt)
                        # state[2] = right_cam (camera, NOT error_energy)
                        # state[3] = velocity
                        # state[4] = heading
                        state_to_evaluate = {
                            "dist": float(state[0]),
                            "tilt": 0.0,
                            "error_energy": 0.0
                        }
                        
                    bypass_active, action_u = self.reflex.evaluate(state_to_evaluate)
                    
                if bypass_active and action_u is not None:
                    cmd_msg = Float64MultiArrayClass()
                    cmd_msg.data = action_u.tolist()
                    self.motor_pub.publish(cmd_msg)
                    
                    telem_msg = Float64MultiArrayClass()
                    telem_msg.data = [1.0, 0.0]
                    self.telemetry_pub.publish(telem_msg)
                else:
                    obs_slices = []
                    expected_len = M_ * slice_dim
                    if data_len >= expected_len:
                        for i in range(M_):
                            obs_slices.append(np.array(msg.data[i*slice_dim : (i+1)*slice_dim]))
                    else:
                        flat_data = np.zeros(expected_len)
                        n_copy = min(data_len, expected_len)
                        flat_data[:n_copy] = msg.data[:n_copy]
                        for i in range(M_):
                            obs_slices.append(flat_data[i*slice_dim : (i+1)*slice_dim])
                    
                    action = Exogenous(np.zeros(2))
                    
                    z, M_val = self.net.step(obs_slices, action, reward=self.reward)
                    
                    action_vector = z[:, 0] if z.ndim > 1 else z
                    vels = self.motor_processor.process(action_vector)
                    
                    cmd_msg = Float64MultiArrayClass()
                    cmd_msg.data = vels.tolist()
                    self.motor_pub.publish(cmd_msg)
                    
                    telem_msg = Float64MultiArrayClass()
                    telem_msg.data = [2.0, float(M_val)] + z.flatten().tolist()
                    self.telemetry_pub.publish(telem_msg)
            except (TypeError, ValueError) as e:
                self.get_logger().error(f"Error processing sensory message: {e}")


# ==========================================
# cerebrum/grounding/reflex.py
# ==========================================


class System1Reflex:
    """Low-latency reactive controller (Cerebellum) bypassing System 2 settling."""
    def __init__(self, collision_threshold=0.20, tilt_threshold=0.5):
        self.collision_threshold = collision_threshold
        self.tilt_threshold = tilt_threshold
        self.last_escape_time = 0.0
        
    def evaluate(self, sensory_state):
        import torch
        if isinstance(sensory_state, (list, tuple, np.ndarray, torch.Tensor)):
            if len(sensory_state) < 3:
                raise ValueError("Sensory state sequence must have at least 3 elements.")
        if isinstance(sensory_state, dict):
            dist = sensory_state.get("dist", 0.0)
            tilt = sensory_state.get("tilt", 0.0)
            error_energy = sensory_state.get("error_energy", 0.0)
        elif hasattr(sensory_state, "dist") or hasattr(sensory_state, "tilt") or hasattr(sensory_state, "error_energy"):
            dist = getattr(sensory_state, "dist", 0.0)
            tilt = getattr(sensory_state, "tilt", 0.0)
            error_energy = getattr(sensory_state, "error_energy", 0.0)
        else:
            dist = sensory_state[0]
            tilt = sensory_state[1]
            error_energy = sensory_state[2]
        
        is_collision_hazard = dist < self.collision_threshold
        is_imbalance_hazard = abs(tilt) > self.tilt_threshold
        is_surprise_hazard = error_energy > 5.0
        
        if is_collision_hazard or is_imbalance_hazard or is_surprise_hazard:
            if is_collision_hazard:
                return True, np.array([0.0, -1.5])  # BACKWARD maneuver
            if is_imbalance_hazard:
                return True, np.array([-1.0, -1.0])  # STABILIZE maneuver
            return True, np.array([0.0, 0.0])  # RE-SETTLE standby
        return False, None


# ==========================================
# cerebrum/grounding/__init__.py
# ==========================================


__all__ = [
    'SensoryProcessor',
    'MotorProcessor',
    'MockPyBullet',
    'MockRclpy',
    'MockNode',
    'MockPublisher',
    'MockSubscription',
    'std_msgs',
    'CerebrumROSNode',
    'System1Reflex'
]
