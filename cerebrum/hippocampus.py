import torch
import numpy as np
from .types import safe_to

class Hippocampus:
    """Explicit Episodic Memory (Hippocampus) using Vector DB / RAG logic.
    Stores one-shot memories as key-value pairs (e.g., key=context latent vector, value=episode detail)
    and retrieves them using cosine similarity."""
    def __init__(self, key_dim, capacity=1000, device='cpu', dtype=torch.float64):
        self.key_dim = key_dim
        self.capacity = capacity
        self.device = device
        self.dtype = dtype
        
        self.keys = torch.zeros((capacity, key_dim), device=device, dtype=dtype)
        self.values = [None] * capacity
        self.timestamps = np.zeros(capacity, dtype=np.int64)
        
        self.size = 0
        self.clock = 0

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.keys = safe_to(self.keys, device, self.dtype)
        return self

    def write(self, key, value):
        """Write a new episodic memory key-value pair. Evicts using LRU policy if full."""
        self.clock += 1
        
        # Convert key to PyTorch tensor on target device
        if isinstance(key, (np.ndarray, list, tuple)):
            key_t = torch.tensor(key, device=self.device, dtype=self.dtype)
        elif isinstance(key, torch.Tensor):
            key_t = key.to(device=self.device, dtype=self.dtype)
        else:
            key_t = torch.as_tensor(key, device=self.device, dtype=self.dtype)
            
        if key_t.dim() > 1:
            key_t = key_t.flatten()
            
        if self.size < self.capacity:
            idx = self.size
            self.size += 1
        else:
            # LRU eviction: find the index with the oldest timestamp
            idx = int(np.argmin(self.timestamps))
            
        self.keys[idx] = key_t
        self.values[idx] = value
        self.timestamps[idx] = self.clock
        return idx

    def retrieve(self, query_key, k=1, threshold=0.0):
        """Retrieve the top-k most similar episodic memories based on cosine similarity."""
        if self.size == 0:
            return []
            
        if isinstance(query_key, (np.ndarray, list, tuple)):
            query_t = torch.tensor(query_key, device=self.device, dtype=self.dtype)
        elif isinstance(query_key, torch.Tensor):
            query_t = query_key.to(device=self.device, dtype=self.dtype)
        else:
            query_t = torch.as_tensor(query_key, device=self.device, dtype=self.dtype)
            
        if query_t.dim() > 1:
            query_t = query_t.flatten()
            
        # Cosine similarity: (keys * query) / (||keys|| * ||query||)
        active_keys = self.keys[:self.size]
        query_norm = torch.linalg.norm(query_t)
        if query_norm == 0.0:
            return []
            
        keys_norm = torch.linalg.norm(active_keys, dim=1)
        # Avoid division by zero
        keys_norm = torch.where(keys_norm == 0.0, torch.ones_like(keys_norm), keys_norm)
        
        sims = (active_keys @ query_t) / (keys_norm * query_norm)
        
        # Sort similarities descending
        vals, indices = torch.sort(sims, descending=True)
        
        results = []
        for i in range(min(k, self.size)):
            sim_val = float(vals[i].item())
            if sim_val >= threshold:
                idx = int(indices[i].item())
                results.append({
                    "similarity": sim_val,
                    "key": active_keys[idx].cpu().numpy(),
                    "value": self.values[idx],
                    "timestamp": int(self.timestamps[idx])
                })
        return results

    def clear(self):
        self.size = 0
        self.clock = 0
        self.keys.zero_()
        self.values = [None] * self.capacity
        self.timestamps.fill(0)
