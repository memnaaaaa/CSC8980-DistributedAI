# strategies.py
# This script defines various compression strategies for federated learning. These strategies are used to
# reduce communication costs by compressing the model updates sent from clients to the server. The strategies
# include no compression (FedAvg), top-k sparsification, sign-based quantization (SignSGD), and uniform
# quantization (QSGD). Additionally, we include a LocalSGD strategy that allows clients to perform more
# local computation before communicating, effectively reducing the frequency of communication without
# changing the actual update size.


import numpy as np


class CompressionStrategy:
    """Base class for compression methods."""
    
    def compress(self, update):
        """Compress model update. Returns compressed form and metadata."""
        return update, {}
    
    def decompress(self, compressed, metadata):
        """Decompress back to full update."""
        return compressed


class NoCompression(CompressionStrategy):
    """Standard FedAvg - no compression."""
    pass


class TopKSparsification(CompressionStrategy):
    """Send only top-k% largest magnitude entries."""
    
    def __init__(self, k_percent=10):
        self.k_percent = k_percent
    
    def compress(self, update):
        """Keep only top k% by magnitude."""
        flat = update.flatten()
        k = max(1, int(len(flat) * self.k_percent / 100))
        
        # Find top-k indices
        threshold = np.partition(np.abs(flat), -k)[-k]
        mask = np.abs(flat) >= threshold
        
        # Sparse representation: (indices, values)
        indices = np.where(mask)[0]
        values = flat[indices]
        
        return (indices, values), {'shape': update.shape, 'total_len': len(flat)}
    
    def decompress(self, compressed, metadata):
        """Reconstruct sparse update."""
        indices, values = compressed
        update = np.zeros(metadata['total_len'])
        update[indices] = values
        return update.reshape(metadata['shape'])


class SignSGD(CompressionStrategy):
    """1-bit quantization: send only sign of each coordinate."""
    
    def compress(self, update):
        """Quantize to {-1, +1}."""
        # Scale by average magnitude for better reconstruction
        scale = np.mean(np.abs(update))
        signs = np.sign(update)
        signs[signs == 0] = 1  # Handle zeros
        
        # Pack bits (simplified: just store as int8)
        return (signs.astype(np.int8), scale), {'shape': update.shape}
    
    def decompress(self, compressed, metadata):
        """Reconstruct using scale."""
        signs, scale = compressed
        return (signs.astype(float) * scale).reshape(metadata['shape'])


class QSGD(CompressionStrategy):
    """Quantized SGD with s levels."""
    
    def __init__(self, num_levels=256):
        self.num_levels = num_levels
    
    def compress(self, update):
        """Uniform quantization."""
        # Normalize
        norm = np.linalg.norm(update)
        if norm == 0:
            return (np.zeros_like(update, dtype=np.int16), 0.0), {'shape': update.shape}
        
        normalized = update / norm
        # Quantize to levels
        levels = np.round((normalized + 1) / 2 * (self.num_levels - 1))
        levels = np.clip(levels, 0, self.num_levels - 1).astype(np.int16)
        
        return (levels, norm), {'shape': update.shape, 'levels': self.num_levels}
    
    def decompress(self, compressed, metadata):
        """Dequantize."""
        levels, norm = compressed
        # Back to [-1, 1]
        normalized = levels / (metadata['levels'] - 1) * 2 - 1
        return (normalized * norm).reshape(metadata['shape'])
    

class LocalSGD(CompressionStrategy):
    """
    Local SGD: clients perform multiple local epochs before communicating.
    This is actually what FedAvg already does, but we make it explicit here
    by varying the number of local steps between synchronizations.
    """
    
    def __init__(self, local_steps=10):
        self.local_steps = local_steps  # Number of SGD steps before sync
    
    def compress(self, update):
        # No actual compression, just fewer communications
        return update, {}
