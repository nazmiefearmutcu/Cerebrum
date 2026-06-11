import torch
from .invariants import assert_one_hot
from .types import to_tensor, safe_to

class Workspace:
    """k<<n latent slots. STRICT one-hot write (slot j content = the single winning module's read);
    soft-weighted aggregation is FORBIDDEN (BAN-1). Broadcast returns slot contents to all modules
    as a top-down prediction (efference copy)."""
    def __init__(self, k_slots, content_dim, device='cpu', dtype=torch.float64):
        self.k = k_slots; self.dim = content_dim
        self.device = device
        self.dtype = dtype
        self.slots = torch.zeros((k_slots, content_dim), device=device, dtype=dtype)

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.slots = safe_to(self.slots, device, self.dtype)
        return self

    def write(self, z, reads):
        assert_one_hot(z, axis=0)                       # raises on soft weights
        z_t = to_tensor(z, self.device, self.dtype)
        reads_t = to_tensor(reads, self.device, self.dtype)
        
        for j in range(self.k):
            winners = torch.nonzero(z_t[:, j] > 0.5, as_tuple=True)[0]
            if winners.numel():
                # reads_t could be a list/tuple of tensors, or a 2D tensor
                self.slots[j] = reads_t[winners[0]]        # strict one-hot read of the winner

    def broadcast(self):
        return self.slots.sum(dim=0)
