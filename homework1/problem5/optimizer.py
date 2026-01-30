# optimizer.py
# This script implements the Gradient Descent optimizer for linear regression from scratch.
# Provides functionality to minimize the Mean Squared Error loss.

# Importing necessary libraries
import numpy as np # for numerical computations
from typing import List, Tuple # for type annotations

# GD Optimizer Class
class LinearRegressionGD:
    """
    Linear regression using Batch Gradient Descent.
    
    Minimizes: L(w) = (1/N) * ||Xw - y||^2
    Gradient: ∇L(w) = (2/N) * X^T(Xw - y)
    """
    
    def __init__(self, step_size: float = 0.01, max_iterations: int = 1000):
        """
        Initialize GD optimizer.
        
        Args:
            step_size: Learning rate α
            max_iterations: Maximum number of gradient steps
        """
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.w_history: List[np.ndarray] = []  # Track parameter trajectory
        self.loss_history: List[float] = []     # Track objective values
        
    def compute_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """Compute MSE loss: (1/N) * ||Xw - y||^2"""
        n_samples = X.shape[0]
        residuals = X @ w - y
        return float(np.mean(residuals ** 2))
    
    def compute_gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Compute gradient of MSE loss.
        
        ∇L(w) = (2/N) * X^T(Xw - y)
        """
        n_samples = X.shape[0]
        residuals = X @ w - y  # Shape: (n_samples,)
        gradient = (2 / n_samples) * (X.T @ residuals)  # Shape: (n_features,)
        return gradient
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        w_init: np.ndarray | None = None,
        w_star: np.ndarray | None = None
    ) -> Tuple[List[float], List[float]]:
        """
        Run gradient descent optimization.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            w_init: Initial parameters. If None, uses zeros.
            w_star: Optimal parameters for tracking ||w_k - w*||. 
                   If provided, tracks parameter error.
                   
        Returns:
            loss_history: Loss values per iteration
            param_error_history: ||w_k - w*|| values per iteration (if w_star provided)
        """
        n_features = X.shape[1]
        
        # Initialize weights
        if w_init is None:
            w = np.zeros(n_features)
        else:
            w = w_init.copy()
            
        self.w_history = [w.copy()]
        self.loss_history = [self.compute_loss(X, y, w)]
        param_errors = []
        
        if w_star is not None:
            param_errors.append(float(np.linalg.norm(w - w_star)))
        
        # Gradient descent iterations
        for iteration in range(self.max_iterations):
            # Compute gradient
            gradient = self.compute_gradient(X, y, w)
            
            # Gradient descent update: w_{k+1} = w_k - α * ∇L(w_k)
            w = w - self.step_size * gradient
            
            # Record metrics
            self.w_history.append(w.copy())
            loss = self.compute_loss(X, y, w)
            self.loss_history.append(loss)
            
            if w_star is not None:
                param_error = float(np.linalg.norm(w - w_star))
                param_errors.append(param_error)
                
        return self.loss_history, param_errors
