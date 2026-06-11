import torch

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
