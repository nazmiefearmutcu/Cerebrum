import numpy as np
import torch
from .invariants import assert_exogenous_action
from .types import to_tensor, safe_to

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
        if getattr(cfg, "non_commutative_prior", False):
            self.R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(M, 1, 1)
        self.parent_vec = None
        self.child_vecs = None
        self.n_nodes = None
        self.B = None
        self.tree_stack = []
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
        if hasattr(self, 'R'):
            self.R = safe_to(self.R, device, self.dtype)
        if self.store is not None:
            self.store = safe_to(self.store, device, self.dtype)
        return self

    def reset(self):
        if getattr(self.cfg, "non_commutative_prior", False):
            self.R = torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(self.cfg.grid_n_modules, 1, 1)
            self.tree_stack = []
        else:
            self._pos = torch.zeros(2, device=self.device, dtype=self.dtype)

    def transition(self, action):
        assert_exogenous_action(action)
        val = to_tensor(action.value, self.device, self.dtype)
        if getattr(self.cfg, "non_commutative_prior", False):
            if self.parent_vec is not None and self.child_vecs is not None and self.n_nodes is not None and self.B is not None:
                # Stack-based path integration on non-metric tree graph
                val_np = val.detach().cpu().numpy() if hasattr(val, "detach") else np.asarray(val)
                def to_np(x):
                    if hasattr(x, "detach"):
                        return x.detach().cpu().numpy()
                    return np.asarray(x)
                
                # Calculate current node index from stack
                curr_node = 0
                for c in self.tree_stack:
                    curr_node = curr_node * self.B + c
                
                parent_np = to_np(self.parent_vec)
                if np.allclose(val_np, parent_np, atol=1e-5):
                    if len(self.tree_stack) > 0:
                        self.tree_stack.pop()
                else:
                    matched = False
                    for idx, child_vec in enumerate(self.child_vecs):
                        child_np = to_np(child_vec)
                        if np.allclose(val_np, child_np, atol=1e-5):
                            c = idx + 1
                            next_node = curr_node * self.B + c
                            if next_node < self.n_nodes:
                                self.tree_stack.append(c)
                            matched = True
                            break
                    if not matched:
                        next_node = curr_node * self.B + 1
                        if next_node < self.n_nodes:
                            self.tree_stack.append(1)

            theta_x = self.k[:, 0] * val[0]
            theta_y = self.k[:, 1] * val[1]
            
            cos_x = torch.cos(theta_x)
            sin_x = torch.sin(theta_x)
            cos_y = torch.cos(theta_y)
            sin_y = torch.sin(theta_y)
            
            Rx = torch.zeros((self.cfg.grid_n_modules, 3, 3), device=self.device, dtype=self.dtype)
            Rx[:, 0, 0] = 1.0
            Rx[:, 1, 1] = cos_x
            Rx[:, 1, 2] = -sin_x
            Rx[:, 2, 1] = sin_x
            Rx[:, 2, 2] = cos_x
            
            Ry = torch.zeros((self.cfg.grid_n_modules, 3, 3), device=self.device, dtype=self.dtype)
            Ry[:, 0, 0] = cos_y
            Ry[:, 0, 2] = sin_y
            Ry[:, 1, 1] = 1.0
            Ry[:, 2, 0] = -sin_y
            Ry[:, 2, 2] = cos_y
            
            R_action = Rx @ Ry
            self.R = self.R @ R_action
            
            with torch.no_grad():
                U, S, Vh = torch.linalg.svd(self.R)
                self.R = U @ Vh
        else:
            self._pos = self._pos + val

    def encode(self):
        if getattr(self.cfg, "non_commutative_prior", False):
            if self.parent_vec is not None and self.child_vecs is not None:
                # Stack-based path integration mapping to a unique coordinate on tree graph
                B = len(self.child_vecs)
                pos = torch.zeros(2, device=self.device, dtype=self.dtype)
                for depth, child_idx in enumerate(self.tree_stack):
                    angle = 2.0 * np.pi * (child_idx - 1) / max(B, 1)
                    dir_vec = torch.tensor([np.cos(angle), np.sin(angle)], device=self.device, dtype=self.dtype)
                    pos = pos + dir_vec * (0.5 ** (depth + 1))
                phase = self.k @ (pos * 100.0)
                cos_phase = torch.cos(phase)
                sin_phase = torch.sin(phase)
                return torch.stack([cos_phase, sin_phase], dim=1).reshape(-1)
            else:
                # Use first two columns of R (6 independent values per module)
                # This captures the full SO(3) state more faithfully than 2 values
                col0 = self.R[:, :, 0]  # (M, 3)
                col1 = self.R[:, :, 1]  # (M, 3)
                vec = torch.cat([col0, col1], dim=1)  # (M, 6)
                return vec.reshape(-1)
        else:
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
