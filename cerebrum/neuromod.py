import numpy as np

class Neuromodulator:
    def __init__(self, cfg, b_T=0.5, a_Pi=2.0, eta0=1.0):
        self.cfg = cfg; self.r_bar = 0.0
        self.b_T, self.a_Pi, self.eta0 = b_T, a_Pi, eta0
    def update(self, reward):
        M = float(reward) - self.r_bar
        self.r_bar += (1.0/self.cfg.tau_r) * (reward - self.r_bar)  # EMA
        return M
    def temperature(self, M):  return self.cfg.T_floor + self.b_T*max(0.0, M)
    def pi_gain(self, M):      return 1.0/(1.0+np.exp(-self.a_Pi*M))
    def eta(self, M):          return self.eta0*max(0.0, M)
    def t_gate(self, M, eps=1e-3): return 1.0/(abs(M)+eps)
