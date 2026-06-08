import numpy as np
def g_act(u):    return np.tanh(u)
def g_deriv(u):  return 1.0 - np.tanh(u)**2   # f = g_act' evaluated at the PRE-activation
