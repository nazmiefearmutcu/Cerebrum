import numpy as np
from .invariants import assert_one_hot

class Workspace:
    """k<<n latent slots. STRICT one-hot write (slot j content = the single winning module's read);
    soft-weighted aggregation is FORBIDDEN (BAN-1). Broadcast returns slot contents to all modules
    as a top-down prediction (efference copy)."""
    def __init__(self, k_slots, content_dim):
        self.k = k_slots; self.dim = content_dim
        self.slots = np.zeros((k_slots, content_dim))
    def write(self, z, reads):
        assert_one_hot(z, axis=0)                       # raises on soft weights
        for j in range(self.k):
            winners = np.flatnonzero(z[:, j] > 0.5)
            if winners.size:
                self.slots[j] = reads[winners[0]]        # strict one-hot read of the winner
    def broadcast(self):
        return self.slots.sum(axis=0)
