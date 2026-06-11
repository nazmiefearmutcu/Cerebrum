import torch
from .types import to_tensor, safe_to

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
