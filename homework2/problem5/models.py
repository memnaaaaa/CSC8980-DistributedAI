# models.py
# This script defines the LogisticRegression model used by clients in federated learning. It
# includes methods for prediction, loss calculation, accuracy, and gradient computation for
# training. The model is a simple multinomial logistic regression using softmax activation.


import numpy as np


class LogisticRegression:
    """Multinomial logistic regression using softmax."""
    
    def __init__(self, input_dim=784, num_classes=10):
        self.W = np.zeros((input_dim, num_classes))  # Weights
        self.b = np.zeros(num_classes)  # Bias
        
    def softmax(self, z):
        """Numerically stable softmax."""
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def predict_proba(self, X):
        """Return class probabilities."""
        logits = X @ self.W + self.b
        return self.softmax(logits)
    
    def predict(self, X):
        """Return class predictions."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def loss(self, X, y):
        """Cross-entropy loss."""
        probs = self.predict_proba(X)
        # One-hot encode y
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(len(y)), y] = 1
        # Cross-entropy
        log_probs = np.log(probs + 1e-8)
        return -np.mean(np.sum(y_onehot * log_probs, axis=1))
    
    def accuracy(self, X, y):
        """Classification accuracy."""
        preds = self.predict(X)
        return np.mean(preds == y)
    
    def get_params(self):
        """Return flattened parameters."""
        return np.concatenate([self.W.flatten(), self.b])
    
    def set_params(self, params):
        """Set parameters from flattened array."""
        w_size = self.W.size
        self.W = params[:w_size].reshape(self.W.shape)
        self.b = params[w_size:]
    
    def get_gradients(self, X, y):
        """Compute gradients for SGD step."""
        n = X.shape[0]
        probs = self.predict_proba(X)
        
        # One-hot labels
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(len(y)), y] = 1
        
        # Gradients
        dW = X.T @ (probs - y_onehot) / n
        db = np.mean(probs - y_onehot, axis=0)
        
        return dW, db
