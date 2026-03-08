# client.py
# This script defines the Client class for federated learning. Each client has its own local data
# and model, and performs local training to compute updates that are sent to the server. The client can
# also apply compression to the updates before sending them.


import numpy as np


class Client:
    """Local client with private data."""
    
    def __init__(self, client_id, data, model, compression):
        self.id = client_id
        self.X, self.y = data
        self.model = model
        self.compression = compression
        
    def local_train(self, epochs, lr, batch_size=32, local_steps=None):
        """
        Perform local training.
        
        Args:
            epochs: Number of epochs to run (or local_steps if specified)
            lr: Learning rate
            batch_size: Batch size
            local_steps: If set, override epochs and run fixed number of steps
        """
        initial_params = self.model.get_params().copy()
        n_samples = len(self.y)
        
        # Determine how many steps to run
        if local_steps is not None:
            total_steps = local_steps
        else:
            total_steps = (n_samples // batch_size) * epochs
        
        step_count = 0
        while step_count < total_steps:
            # Shuffle data each "epoch"
            indices = np.random.permutation(n_samples)
            
            for i in range(0, n_samples, batch_size):
                if step_count >= total_steps:
                    break
                
                batch_idx = indices[i:i+batch_size]
                X_batch = self.X[batch_idx]
                y_batch = self.y[batch_idx]
                
                # SGD step
                dW, db = self.model.get_gradients(X_batch, y_batch)
                self.model.W -= lr * dW
                self.model.b -= lr * db
                
                step_count += 1
        
        # Compute update
        new_params = self.model.get_params()
        update = new_params - initial_params
        
        compressed, metadata = self.compression.compress(update)
        
        return {
            'update': compressed,
            'metadata': metadata,
            'n_samples': n_samples,
            'steps': step_count
        }
    
    def set_model(self, params):
        """Set global model parameters."""
        self.model.set_params(params)
