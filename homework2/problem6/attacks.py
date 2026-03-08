# attacks.py
# This script defines various attack strategies that malicious clients can use in federated learning. These
# attacks include Gaussian noise injection, Byzantine gradient manipulation, and label flipping. Each attack
# is implemented as a class that can generate malicious updates based on the true update, the global model,
# and the client's local data. The attacks are designed to test the robustness of different aggregation rules
# against Byzantine clients.


import numpy as np


class Attack:
    """Base class for attacks."""
    
    def generate_update(self, true_update, global_model, client_data):
        """Generate malicious update."""
        raise NotImplementedError


class BenignAttack(Attack):
    """No attack—honest client behavior."""
    
    def generate_update(self, true_update, global_model, client_data):
        return true_update


class GaussianAttack(Attack):
    """Send random Gaussian noise instead of true gradient."""
    
    def __init__(self, scale=10.0):
        self.scale = scale
    
    def generate_update(self, true_update, global_model, client_data):
        """Replace update with random Gaussian noise."""
        return np.random.normal(0, self.scale, size=true_update.shape)


class ByzantineGradientAttack(Attack):
    """Send gradient with extremely large magnitude (opposite direction)."""
    
    def __init__(self, scale=100.0):
        self.scale = scale
    
    def generate_update(self, true_update, global_model, client_data):
        """Negate and amplify the gradient."""
        return -self.scale * true_update


class LabelFlippingAttack(Attack):
    """Train on corrupted labels (data poisoning)."""
    
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
    
    def corrupt_labels(self, y):
        """Flip labels: y -> (y + 1) % num_classes."""
        return (y + 1) % self.num_classes
