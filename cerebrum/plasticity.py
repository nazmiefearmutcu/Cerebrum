import numpy as np
import torch
from .types import to_tensor, safe_to

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
            self.value += (1.0/max(self.cfg.tau_e, 1e-6))*(a_pre_t - self.value)
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
        val = Pi_t + (1.0 / max(cfg.tau_pi, 1e-6)) * dPi
        val = torch.clamp(val, min=1e-6)
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

def feedback_update_kp(B, M, Pi_post, eps_post, elig, eta, lam_kp, theta=None):
    """Kolen-Pollack feedback-alignment rule (OPT-IN). Drives B[l] (shape (out_up, in_post.T)
    i.e. (d[l+1], d[l])) toward W[l].T using the SAME local four-factor product that updates
    W[l] -- only TRANSPOSED and gated by the transposed metaplastic consolidation tensor theta --
    plus a MATCHED symmetric weight decay.
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
        if theta is not None:
            theta_t = to_tensor(theta, device, dtype)
            val = eta * M_t * theta_t.t() * torch.outer(pre, post) - lam_kp * B_t
        else:
            val = eta * M_t * torch.outer(pre, post) - lam_kp * B_t
    return val
