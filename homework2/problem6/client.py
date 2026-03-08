# client.py
# This script defines the Client class for federated learning. Each client has its own local data and model,
# and performs local training to compute updates that are sent to the server. The client can also apply
# malicious attacks to test the robustness of the aggregation rules on the server side.


import numpy as np


class Client:
    """Local client with private data and potential malicious behavior."""
    
    def __init__(self, client_id, data, model, attack=None):
        self.id = client_id
        self.X, self.y = data
        self.model = model
        self.attack = attack if attack else None
        
    def local_train(self, epochs, lr, batch_size=32):
        """
        Perform local training.
        Returns: update (delta), number of samples, and whether malicious.
        """
        initial_params = self.model.get_params().copy()
        n_samples = len(self.y)
        
        # Check if this is a label flipping attack (data poisoning)
        y_train = self.y.copy()
        if self.attack and hasattr(self.attack, 'corrupt_labels'):
            y_train = self.attack.corrupt_labels(y_train)
        
        # Local SGD
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = self.X[batch_idx]
                y_batch = y_train[batch_idx]
                
                dW, db = self.model.get_gradients(X_batch, y_batch)
                self.model.W -= lr * dW
                self.model.b -= lr * db
        
        # Compute true update
        new_params = self.model.get_params()
        true_update = new_params - initial_params
        
        # Apply attack if malicious
        is_malicious = self.attack is not None
        if is_malicious:
            final_update = self.attack.generate_update(
                true_update, 
                {'W': self.model.W, 'b': self.model.b},
                (self.X, self.y)
            )
        else:
            final_update = true_update
        
        return final_update, n_samples, is_malicious
    
    def set_model(self, params):
        """Set global model parameters."""
        self.model.set_params(params)
