import numpy as np

class Eligibility:
    """Synapse-local presynaptic low-pass trace: tau_e de/dt = -e + a_pre (bare, no Pi inside)."""
    def __init__(self, shape, cfg):
        self.value = np.zeros(shape); self.cfg = cfg
    def step(self, a_pre):
        self.value += (1.0/self.cfg.tau_e)*(a_pre - self.value)
        return self.value

def weight_update(M, theta, Pi_post, eps_post, elig, eta):
    """Four-factor local Hebbian = eta * M * theta * (Pi_post*eps_post) outer elig.
    Equals eta * (-dF/dW) when M=theta=1 (precision-once convention). theta is (out,in)."""
    post = (Pi_post * eps_post)[:, None]          # (out,1)
    pre = elig[None, :]                           # (1,in)
    return eta * M * theta * (post @ pre)

def precision_update(Pi, eps_sq, cfg):
    """Diagonal, local-per-unit precision learning. Relaxes Pi toward its spec fixed point
    Pi -> 1/(sigma0^2 + <eps^2>) (spec eq. precision learning). kappa_pi sets the relaxation
    gain, tau_pi the timescale; sigma0 is the precision-floor variance. Local: each unit i
    only reads its own eps_i^2 and Pi_i, no cross-unit / matrix-inverse term."""
    target = 1.0 / np.maximum(cfg.sigma0**2 + eps_sq, 1e-6)   # 1/(sigma0^2 + <eps^2>)
    dPi = cfg.kappa_pi * (target - Pi)
    return Pi + (1.0/cfg.tau_pi)*dPi

def feedback_update(B, a_up, eps, cfg):
    """Local feedback-weight rule: eta_b * a_up outer eps - lam_b * B. No transpose of W is read."""
    return cfg.eta_b*np.outer(a_up, eps) - cfg.lam_b*B
