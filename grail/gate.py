import numpy as np
from .invariants import assert_one_hot, assert_scalar_M

class BasalGangliaGate:
    """Stochastic basal-ganglia gate. Modules bid a SCALAR own-error salience for k workspace slots;
    a striatal Go/NoGo competition selects a strict one-hot winner per slot WITH noise (never argmax,
    never soft). Gate weights learn by a LOCAL three-factor rule gated by the scalar neuromodulator M.
    There is NO query-key / content-similarity term anywhere — the competition can never become attention."""
    def __init__(self, n_modules, k_slots, cfg, seed=0):
        self.M_ = n_modules; self.k = k_slots; self.cfg = cfg
        rng = np.random.default_rng(seed + 31)
        self.G = 0.5 + 0.1*rng.standard_normal((n_modules, k_slots))   # Go weights
        self.N = 0.1*rng.standard_normal((n_modules, k_slots))         # NoGo weights
        self.theta = np.zeros(n_modules)                               # dead-expert excitability
        self._P = None; self._z = None; self._bid = None

    def bid(self, err_sq, pi):
        return pi*np.asarray(err_sq) + self.theta                       # (M,) scalar per module

    def select(self, bids, rng, T_gate):
        bids = np.asarray(bids, float)
        z = np.zeros((self.M_, self.k)); P = np.zeros((self.M_, self.k))
        for j in range(self.k):
            inhib_total = float(np.sum(self.N[:, j]*bids))
            u = self.G[:, j]*bids - (inhib_total - self.N[:, j]*bids)   # u_mj = G b_m - sum_{m'!=m} N b_m'
            logits = u/max(T_gate,1e-6) + rng.gumbel((self.M_,))
            ex = np.exp(logits - logits.max()); P[:, j] = ex/ex.sum()
            z[int(np.argmax(logits)), j] = 1.0                          # Gumbel-argmax = exact softmax SAMPLE
        assert_one_hot(z, axis=0)
        self._P, self._z, self._bid = P, z, bids
        return z

    def learn(self, M, eta=None):
        assert_scalar_M(M)
        eta = self.cfg.eta_w if eta is None else eta
        e = (self._z - self._P) * self._bid[:, None]                    # local 3-factor eligibility
        self.G += eta*M*e
        self.N += -eta*M*e                                              # NoGo opponent (opposite sign)
        # Synaptic weight decay toward init (local homeostatic regularization). Without a STABLE
        # per-module target, reward-driven Go drift learns spurious preferences that override the
        # informative scalar bid; decay keeps the gate "trusting the bid" unless reward consistently
        # supports a preference. lam_g=0 -> off (unchanged behavior).
        if self.cfg.lam_g > 0.0:
            self.G += self.cfg.lam_g * (0.5 - self.G)                   # init Go mean
            self.N += self.cfg.lam_g * (0.0 - self.N)                   # init NoGo mean

    def homeostasis(self, M=None, gamma_up=0.02, gamma_dn=0.05):
        """Dead-expert load balancing as per-neuron metabolic homeostasis: excitability theta rises for
        modules that win nothing (anti-dead-expert) and falls for winners (anti-hog). When the scalar
        neuromodulator M is supplied the win-penalty is REWARD-AWARE — a REWARDED win (M>0, i.e. correct
        routing) is NOT treated as hogging — so homeostasis stops fighting correct routing (spec FM5b).
        M is a scalar (BAN-2 safe); M=None keeps the plain anti-hog behavior."""
        wins = np.minimum(self._z.sum(axis=1), 1.0)
        if M is None:
            hog = 1.0
        else:
            assert_scalar_M(M)
            hog = 1.0/(1.0 + np.exp(2.0*float(M)))     # M>0 (rewarded) -> ~0 penalty; M<=0 (hog) -> ~1
        self.theta += gamma_up*(1.0 - wins) - gamma_dn*wins*hog         # rises on loss, falls on un-rewarded win
