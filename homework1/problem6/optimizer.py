# homework1/problem6/optimizer.py
# This script implements Full Gradient Descent, Mini-batch SGD, and Pure SGD optimizers.

# Importing necessary libraries
import numpy as np # for numerical computations
import time # for timing

class GradientDescent:
    """Full-batch GD from Problem 5."""
    def __init__(self, step_size=0.4, max_iter=200):
        self.step_size = step_size
        self.max_iter = max_iter
        self.loss_history = []
        self.time_history = []
    
    def fit(self, X, y, w_star=None):
        n, d = X.shape
        w = np.zeros(d)
        self.loss_history = []
        self.time_history = []
        
        for _ in range(self.max_iter):
            start = time.time()
            
            # Full gradient
            grad = (2/n) * X.T @ (X @ w - y)
            w -= self.step_size * grad
            
            self.time_history.append(time.time() - start)
            loss = np.mean((X @ w - y)**2)
            self.loss_history.append(loss)
        
        return w, self.loss_history


class MiniBatchSGD:
    """Mini-batch SGD."""
    def __init__(self, step_size=0.05, max_iter=200, batch_size=32, seed=42):
        self.step_size = step_size
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.loss_history = []
        self.time_history = []
    
    def fit(self, X, y, w_star=None):
        n, d = X.shape
        w = np.zeros(d)
        self.loss_history = []
        self.time_history = []
        
        iter_count = 0
        while iter_count < self.max_iter:
            # Shuffle each epoch
            idx = self.rng.permutation(n)
            X_shuf, y_shuf = X[idx], y[idx]
            
            for i in range(0, n, self.batch_size):
                if iter_count >= self.max_iter:
                    break
                
                start = time.time()
                
                # Mini-batch
                batch_x = X_shuf[i:i+self.batch_size]
                batch_y = y_shuf[i:i+self.batch_size]
                b = len(batch_y)
                
                grad = (2/b) * batch_x.T @ (batch_x @ w - batch_y)
                w -= self.step_size * grad
                
                self.time_history.append(time.time() - start)
                loss = np.mean((X @ w - y)**2)
                self.loss_history.append(loss)
                iter_count += 1
        
        return w, self.loss_history


class PureSGD:
    """SGD with batch_size=1."""
    def __init__(self, step_size=0.005, max_iter=2000, seed=42):
        self.step_size = step_size
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)
        self.loss_history = []
        self.time_history = []
    
    def fit(self, X, y, w_star=None):
        n, d = X.shape
        w = np.zeros(d)
        self.loss_history = []
        self.time_history = []
        
        iter_count = 0
        while iter_count < self.max_iter:
            idx = self.rng.permutation(n)
            
            for i in idx:
                if iter_count >= self.max_iter:
                    break
                
                start = time.time()
                
                # Single sample gradient: 2*(w*x_i - y_i)*x_i
                xi, yi = X[i], y[i]
                grad = 2 * (np.dot(xi, w) - yi) * xi
                w -= self.step_size * grad
                
                self.time_history.append(time.time() - start)
                
                # Record full loss every 10 steps (for speed)
                if iter_count % 10 == 0:
                    loss = np.mean((X @ w - y)**2)
                    self.loss_history.append(loss)
                
                iter_count += 1
        
        # Pad loss history to match iteration count if needed
        while len(self.loss_history) < self.max_iter // 10:
            self.loss_history.append(np.mean((X @ w - y)**2))
            
        return w, self.loss_history