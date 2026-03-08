# server.py
# This script defines the FederatedServer class, which orchestrates the federated learning process. The server
# aggregates updates from clients, applies the chosen compression strategy, and updates the global model.


import numpy as np


class FederatedServer:
    """Central server orchestrating training."""
    
    def __init__(self, model, compression):
        self.global_model = model
        self.compression = compression
        self.total_bits_sent = 0
        
    def aggregate(self, client_updates):
        """
        FedAvg aggregation with compression handling.
        client_updates: list of dicts from clients
        """
        total_samples = sum(u['n_samples'] for u in client_updates)
        
        # Weighted average of updates
        aggregated_update = np.zeros_like(self.global_model.get_params())
        
        for update_info in client_updates:
            # Decompress
            compressed = update_info['update']
            metadata = update_info['metadata']
            update = self.compression.decompress(compressed, metadata)
            
            # Weight by number of samples
            weight = update_info['n_samples'] / total_samples
            aggregated_update += weight * update
            
            # Track communication cost (simplified)
            if isinstance(compressed, tuple):
                # Sparse or quantized
                self.total_bits_sent += self._estimate_bits(compressed)
            else:
                # Dense float32
                self.total_bits_sent += len(update) * 32
        
        # Apply update to global model
        current_params = self.global_model.get_params()
        new_params = current_params + aggregated_update
        self.global_model.set_params(new_params)
        
    def _estimate_bits(self, compressed):
        """Estimate bits for compressed representation."""
        if isinstance(compressed[0], np.ndarray) and compressed[0].dtype == np.int8:
            # SignSGD: 1 bit per coordinate (approx)
            return len(compressed[0]) * 1 + 32  # + scale
        elif isinstance(compressed[0], tuple):
            # Top-k: indices + values
            indices, values = compressed
            return len(indices) * (32 + 32)  # index + value
        else:
            # QSGD
            levels, _ = compressed
            return len(levels) * 16  # int16
        
    def get_model_params(self):
        return self.global_model.get_params()
    
    def distribute(self, clients):
        """Send global model to all clients."""
        params = self.get_model_params()
        for client in clients:
            client.set_model(params)
