import numpy as np
import torch

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
