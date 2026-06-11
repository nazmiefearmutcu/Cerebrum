import sys
import time
import numpy as np
import torch
import pytest

# Helper to dynamically select target dtype based on device (MPS does not support float64)
def get_device_dtype(device):
    if device == "mps" or (isinstance(device, torch.device) and device.type == "mps"):
        return torch.float32
    return torch.float64

def to_tensor_clean(x, dtype=None, device="cpu"):
    target_dtype = get_device_dtype(device)
    if isinstance(x, torch.Tensor):
        return x.detach().clone().to(dtype=target_dtype, device=device)
    if isinstance(x, (list, tuple)):
        return torch.stack([to_tensor_clean(item, target_dtype, device) for item in x]) if len(x) > 0 else torch.zeros(0, dtype=target_dtype, device=device)
    return torch.tensor(x, dtype=target_dtype, device=device)

def zeros_clean(shape, device="cpu"):
    return torch.zeros(shape, dtype=get_device_dtype(device), device=device)

def ones_clean(shape, device="cpu"):
    return torch.ones(shape, dtype=get_device_dtype(device), device=device)

# ========================================== GROUNDING PACKAGE IMPORTS ==========================================
from cerebrum.grounding import (
    SensoryProcessor,
    MotorProcessor,
    MockPyBullet,
    MockRclpy,
    MockNode,
    std_msgs
)

# ========================================== SYSTEM 1 REFLEX BYPASS ==========================================
from cerebrum.grounding import System1Reflex


# ========================================== PYTORCH BACKEND IMPLEMENTATION ==========================================
class TorchPCAreas:
    def __init__(self, np_pc, device="cpu"):
        self.cfg = np_pc.cfg
        self.L = np_pc.L
        self.device = device
        self.x = [to_tensor_clean(xl, None, device) for xl in np_pc.x]
        self.eps = [to_tensor_clean(el, None, device) for el in np_pc.eps]
        self.Pi = [to_tensor_clean(pl, None, device) for pl in np_pc.Pi]
        self.W = [to_tensor_clean(wl, None, device) for wl in np_pc.W]
        self.B = [to_tensor_clean(bl, None, device) for bl in np_pc.B]

    def _bottomup_scale_top(self):
        if self.L < 2:
            return torch.norm(self.x[-1]).item()
        fprime = 1.0 - torch.tanh(torch.matmul(self.W[self.L-2], self.x[self.L-1]))**2
        fb = torch.matmul(self.B[self.L-2], fprime * (self.Pi[self.L-2] * self.eps[self.L-2]))
        s = torch.norm(fb).item()
        if s == 0.0:
            s = torch.norm(self.x[-1]).item()
        return s

    def _balanced_top_pred(self, top_pred):
        if top_pred is None or not getattr(self.cfg, "balance_grid_precision", False):
            return top_pred
        pnorm = torch.norm(top_pred).item()
        if pnorm == 0.0:
            return top_pred
        ref = getattr(self.cfg, "grid_precision_ref", 1.0) * self._bottomup_scale_top()
        scale = min(1.0, ref / pnorm)
        return top_pred * scale

    def predict(self, l, top_pred=None):
        if l < self.L-1:
            return torch.tanh(torch.matmul(self.W[l], self.x[l+1]))
        if top_pred is None:
            return zeros_clean(self.x[l].shape, self.device)
        return self._balanced_top_pred(top_pred)

    def compute_errors(self, top_pred=None, broadcast=None):
        for l in range(self.L):
            yhat = self.predict(l, top_pred=top_pred)
            p = 0.0 if broadcast is None or broadcast[l] is None else broadcast[l]
            self.eps[l] = self.x[l] - yhat - p

    def settle_step(self, rng, T, clamp_bottom=None, top_pred=None, broadcast=None, counters=None):
        self.compute_errors(top_pred=top_pred, broadcast=broadcast)
        c = self.cfg
        new_x = [xl.clone() for xl in self.x]
        for l in range(self.L):
            if l == 0 and clamp_bottom is not None:
                new_x[0] = clamp_bottom.clone()
                continue
            drift = -self.Pi[l] * self.eps[l]
            if l >= 1:
                fprime = 1.0 - torch.tanh(torch.matmul(self.W[l-1], self.x[l]))**2
                drift = drift + torch.matmul(self.B[l-1], fprime * (self.Pi[l-1] * self.eps[l-1]))
            drift = drift - c.gamma * torch.sign(self.x[l])
            step = (drift / c.tau_x) * c.dt
            
            noise_np = rng.normal(self.x[l].shape, scale=np.sqrt(2.0*T*c.dt/c.tau_x))
            noise = to_tensor_clean(noise_np, None, self.device)
            new_x[l] = self.x[l] + step + noise
            if l >= 1 and getattr(self.cfg, "pc_sparsity_threshold", 0.0) > 0.0:
                new_x[l] = torch.where(torch.abs(new_x[l]) < self.cfg.pc_sparsity_threshold, zeros_clean(new_x[l].shape, self.device), new_x[l])
        
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
            
        self.x = new_x
        if counters is not None:
            for xl in self.x[1:]:
                counters.record_activity(xl.cpu().numpy())

class TorchGridHead:
    def __init__(self, np_gh, device="cpu"):
        self.cfg = np_gh.cfg
        self.device = device
        self.k = to_tensor_clean(np_gh.k, None, device)
        self.pos = to_tensor_clean(np_gh.pos, None, device)
        if np_gh.store is not None:
            self.store = to_tensor_clean(np_gh.store, None, device)
        else:
            self.store = None
        self.obs_dim = np_gh.obs_dim

    def transition(self, action_val):
        self.pos = self.pos + action_val

    def encode(self):
        phase = torch.matmul(self.k, self.pos)
        return torch.stack([torch.cos(phase), torch.sin(phase)], dim=1).reshape(-1)

    def bind(self, obs, M=1.0):
        self._ensure_store(obs.size(0))
        g = self.encode()
        self.store += self.cfg.grid_eta_bind * M * torch.outer(obs, g)

    def _ensure_store(self, obs_dim):
        if self.store is None:
            self.obs_dim = obs_dim
            self.store = zeros_clean((obs_dim, self.encode().size(0)), self.device)

    def complete(self):
        g = self.encode()
        return torch.matmul(self.store, g) if self.store is not None else zeros_clean(self.obs_dim or 1, self.device)

class TorchBasalGangliaGate:
    def __init__(self, np_bg, device="cpu"):
        self.M_ = np_bg.M_
        self.k = np_bg.k
        self.cfg = np_bg.cfg
        self.device = device
        self.G = to_tensor_clean(np_bg.G, None, device)
        self.N = to_tensor_clean(np_bg.N, None, device)
        self.theta = to_tensor_clean(np_bg.theta, None, device)
        self._P = None
        self._z = None
        self._bid = None

    def bid(self, err_sq, pi):
        return pi * err_sq + self.theta

    def select(self, bids, rng, T_gate):
        z = zeros_clean((self.M_, self.k), self.device)
        P = zeros_clean((self.M_, self.k), self.device)
        for j in range(self.k):
            inhib_total = torch.sum(self.N[:, j] * bids)
            u = self.G[:, j] * bids - (inhib_total - self.N[:, j] * bids)
            gumbel_np = rng.gumbel((self.M_,))
            gumbel_noise = to_tensor_clean(gumbel_np, None, self.device)
            logits = u / max(T_gate, 1e-6) + gumbel_noise
            ex = torch.exp(logits - logits.max())
            P[:, j] = ex / ex.sum()
            z[torch.argmax(logits).item(), j] = 1.0
        self._P, self._z, self._bid = P, z, bids
        return z

    def learn(self, M, eta=None):
        eta = self.cfg.eta_w if eta is None else eta
        e = (self._z - self._P) * self._bid[:, None]
        self.G += eta * M * e
        self.N += -eta * M * e
        if self.cfg.lam_g > 0.0:
            self.G += self.cfg.lam_g * (0.5 - self.G)
            self.N += self.cfg.lam_g * (0.0 - self.N)

    def homeostasis(self, M=None, gamma_up=0.02, gamma_dn=0.05):
        wins = torch.minimum(self._z.sum(dim=1), ones_clean(self.M_, self.device))
        if M is None:
            hog = 1.0
        else:
            hog = 1.0 / (1.0 + torch.exp(2.0 * M))
        self.theta += gamma_up * (1.0 - wins) - gamma_dn * wins * hog

class TorchWorkspace:
    def __init__(self, np_w, device="cpu"):
        self.k = np_w.k
        self.dim = np_w.dim
        self.device = device
        self.slots = to_tensor_clean(np_w.slots, None, device)

    def write(self, z, reads):
        for j in range(self.k):
            winners = torch.nonzero(z[:, j] > 0.5, as_tuple=True)[0]
            if winners.numel():
                self.slots[j] = reads[winners[0].item()]

    def broadcast(self):
        return self.slots.sum(dim=0)

class TorchEligibility:
    def __init__(self, np_e, device="cpu"):
        self.cfg = np_e.cfg
        self.device = device
        self.value = to_tensor_clean(np_e.value, None, device)

    def step(self, a_pre):
        self.value += (1.0 / self.cfg.tau_e) * (a_pre - self.value)
        return self.value

class TorchMetaplasticFuse:
    def __init__(self, np_f, device="cpu"):
        self.cfg = np_f.cfg
        self.device = device
        self.c = to_tensor_clean(np_f.c, None, device)
        self.S_bar = to_tensor_clean(np_f.S_bar, None, device)

    def _raw_surprise(self, Pi_post, eps_post, elig):
        return torch.abs((Pi_post * eps_post)[:, None] * elig[None, :])

    def update(self, Pi_post, eps_post, elig):
        S_raw = self._raw_surprise(Pi_post, eps_post, elig)
        S = S_raw - self.S_bar
        predictive = (S_raw <= self.S_bar).to(dtype=get_device_dtype(self.device))
        surprising = torch.clamp(S, min=0.0)
        dc = self.cfg.alpha_c * predictive * (self.cfg.c_max - self.c) - self.cfg.beta_c * surprising * self.c
        self.c = torch.clamp(self.c + (1.0 / self.cfg.tau_c) * dc, 0.0, self.cfg.c_max)
        self.S_bar += (1.0 / self.cfg.tau_S) * (S_raw - self.S_bar)
        theta = 1.0 / (1.0 + torch.exp(-self.cfg.g_theta * (S - self.c)))
        return theta

class TorchNeuromodulator:
    def __init__(self, np_nm, device="cpu"):
        self.cfg = np_nm.cfg
        self.device = device
        self.r_bar = np_nm.r_bar
        self.b_T = np_nm.b_T
        self.a_Pi = np_nm.a_Pi
        self.eta0 = np_nm.eta0

    def update(self, reward):
        M = float(reward) - self.r_bar
        self.r_bar += (1.0 / self.cfg.tau_r) * (reward - self.r_bar)
        return torch.tensor(M, dtype=get_device_dtype(self.device), device=self.device)

    def temperature(self, M):
        m_val = float(M)
        return self.cfg.T_floor + self.b_T * max(0.0, m_val)

    def pi_gain(self, M):
        m_val = float(M)
        return 1.0 / (1.0 + np.exp(-self.a_Pi * m_val))

    def eta(self, M):
        m_val = float(M)
        return self.eta0 * max(0.0, m_val)

    def t_gate(self, M, eps=1e-3):
        m_val = float(M)
        return 1.0 / (abs(m_val) + eps)

class TorchCerebrumNet:
    def __init__(self, np_net, device="cpu"):
        self.cfg = np_net.cfg
        self.M_ = np_net.M_
        self.k = np_net.k
        self.content_dim = np_net.content_dim
        self.device = device
        
        self.modules = [TorchPCAreas(m, device) for m in np_net.modules]
        self.grid = TorchGridHead(np_net.grid, device)
        self.gate = TorchBasalGangliaGate(np_net.gate, device)
        self.workspace = TorchWorkspace(np_net.workspace, device)
        self.nm = TorchNeuromodulator(np_net.nm, device)
        self.rng = np_net.rng
        self.counters = np_net.counters
        
        self.elig = [[TorchEligibility(elig_layer, device) for elig_layer in elig_mod] for elig_mod in np_net.elig]
        self.fuse = [[TorchMetaplasticFuse(fuse_layer, device) for fuse_layer in fuse_mod] for fuse_mod in np_net.fuse]
        
        self._force_theta = np_net._force_theta
        self.last_theta = None
        self.last_top_pred = zeros_clean(self.content_dim, device)
        self._U = None if getattr(np_net, "_U", None) is None else to_tensor_clean(np_net._U, None, device)

    def _top_pred_from_grid(self, obs_dim):
        rec = self.grid.complete() if self.grid.store is not None else zeros_clean(obs_dim, self.device)
        if self._U is None:
            rng = np.random.default_rng(self.cfg.seed + 7)
            np_U = 0.1 * rng.standard_normal((self.content_dim, obs_dim))
            self._U = to_tensor_clean(np_U, None, self.device)
        self.counters.record_global_infer_vectors(k=1, width=self.content_dim)
        return torch.matmul(self._U, rec)

    def _broadcast_for_module(self, mod, wksp):
        b = [None] * mod.L
        d0 = mod.cfg.dims[0]
        p0 = zeros_clean(d0, self.device)
        n = min(d0, wksp.size(0))
        p0[:n] = wksp[:n]
        b[0] = p0
        return b

    def _settle_all(self, obs_slices, top_pred, wksp, T, learn=False):
        err_sq = zeros_clean(self.M_, self.device)
        reads = zeros_clean((self.M_, self.content_dim), self.device)
        for m_i, mod in enumerate(self.modules):
            bcast = self._broadcast_for_module(mod, wksp)
            obs_tensor = to_tensor_clean(obs_slices[m_i], None, self.device)
            for _ in range(self.cfg.n_settle):
                mod.settle_step(self.rng, T=T, clamp_bottom=obs_tensor,
                                top_pred=top_pred, broadcast=bcast, counters=self.counters)
                if learn:
                    for l in range(mod.L - 1):
                        self.elig[m_i][l].step(a_pre=mod.x[l + 1])
            mod.compute_errors(top_pred=top_pred, broadcast=bcast)
            err_sq[m_i] = sum(torch.sum(e ** 2) for e in mod.eps)
            reads[m_i] = mod.x[-1].clone()
        return err_sq, reads

    def settle_only(self, obs_slices, action, T=None):
        action_val = to_tensor_clean(action.value, None, self.device)
        self.grid.transition(action_val)
        top_pred = self._top_pred_from_grid(self.modules[0].cfg.dims[0])
        self.last_top_pred = top_pred.clone()
        wksp = self.workspace.broadcast()
        T = self.nm.temperature(0.0) if T is None else T
        err_sq, reads = self._settle_all(obs_slices, top_pred, wksp, T)
        return err_sq.cpu().numpy(), reads.cpu().numpy()

    def step(self, obs_slices, action, reward):
        action_val = to_tensor_clean(action.value, None, self.device)
        self.grid.transition(action_val)
        top_pred = self._top_pred_from_grid(self.modules[0].cfg.dims[0])
        self.last_top_pred = top_pred.clone()
        
        M_preview = float(reward) - self.nm.r_bar
        obs_mean_np = np.mean(np.stack([np.asarray(o, float) for o in obs_slices]), axis=0)
        obs_mean = to_tensor_clean(obs_mean_np, None, self.device)
        self.grid.bind(obs_mean, M=max(M_preview, 0.0))

        wksp = self.workspace.broadcast()
        self.counters.record_global_infer_vectors(k=self.k, width=self.content_dim)
        T = self.nm.temperature(0.0)
        err_sq, reads = self._settle_all(obs_slices, top_pred, wksp, T, learn=True)

        pi = torch.stack([mod.Pi[-1].mean() for mod in self.modules])
        bids = self.gate.bid(err_sq=err_sq, pi=pi)
        T_gate = self.cfg.gate_temp if self.cfg.gate_temp > 0.0 else self.nm.t_gate(max(reward, 1e-3))
        
        z = self.gate.select(bids, self.rng, T_gate=T_gate)
        self.workspace.write(z, reads)

        M = self.nm.update(reward)
        self.counters.record_global_learn(1)
        self.last_theta = [[None] * (mod.L - 1) for mod in self.modules]
        
        for m_i, mod in enumerate(self.modules):
            for l in range(mod.L - 1):
                theta = self.fuse[m_i][l].update(mod.Pi[l], mod.eps[l], self.elig[m_i][l].value)
                if self._force_theta is not None:
                    theta = torch.full_like(mod.W[l], float(self._force_theta))
                self.last_theta[m_i][l] = theta
                
                post = (mod.Pi[l] * mod.eps[l])[:, None]
                pre = self.elig[m_i][l].value[None, :]
                dW = (self.cfg.eta_w / self.cfg.tau_w) * M * theta * torch.matmul(post, pre)
                mod.W[l] += dW
                
                a_up = mod.x[l + 1]
                eps = mod.eps[l]
                dB = self.cfg.eta_b * torch.outer(a_up, eps) - self.cfg.lam_b * mod.B[l]
                mod.B[l] += (1.0 / self.cfg.tau_b) * dB
                
                eps_sq = mod.eps[l] ** 2
                target = 1.0 / torch.clamp(self.cfg.sigma0**2 + eps_sq, min=1e-6)
                dPi = self.cfg.kappa_pi * (target - mod.Pi[l])
                mod.Pi[l] = mod.Pi[l] + (1.0 / self.cfg.tau_pi) * dPi
                
        self.gate.learn(M=M)
        self.gate.homeostasis(M=M)
        return z.cpu().numpy(), M

def patch_cerebrum_net():
    from cerebrum.unified import CerebrumNet
    
    if hasattr(CerebrumNet, "set_backend"):
        return
        
    original_init = CerebrumNet.__init__
    
    def new_init(self, n_modules, k_slots, slice_dim, cfg):
        original_init(self, n_modules, k_slots, slice_dim, cfg)
        self._backend = "numpy"
        self._device = "cpu"
        self._torch_net = None
        
    def set_backend(self, backend, device="cpu"):
        self._backend = backend
        self._device = device
        if backend == "torch":
            self._torch_net = TorchCerebrumNet(self, device=device)
            # Link components directly so inspecting net.modules works
            self.modules = self._torch_net.modules
            self.elig = self._torch_net.elig
            self.fuse = self._torch_net.fuse
            self.grid = self._torch_net.grid
            self.gate = self._torch_net.gate
            self.workspace = self._torch_net.workspace
            self.nm = self._torch_net.nm
            
    def step_patched(self, obs_slices, action, reward):
        if self._backend == "torch":
            return self._torch_net.step(obs_slices, action, reward)
        else:
            return self.step_original(obs_slices, action, reward)
            
    def settle_only_patched(self, obs_slices, action, T=None):
        if self._backend == "torch":
            return self._torch_net.settle_only(obs_slices, action, T)
        else:
            return self.settle_only_original(obs_slices, action, T)
            
    CerebrumNet.__init__ = new_init
    CerebrumNet.set_backend = set_backend
    CerebrumNet.step_original = CerebrumNet.step
    CerebrumNet.step = step_patched
    CerebrumNet.settle_only_original = CerebrumNet.settle_only
    CerebrumNet.settle_only = settle_only_patched
