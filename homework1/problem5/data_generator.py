# data_generator.py
# This script handles synthetic data generation for linear regression.
# Provides ground-truth data with controlled noise levels.

# Importing necessary libraries
import numpy as np # for numerical computations
from typing import Tuple # for type annotations

# Function to generate synthetic linear regression data
def generate_synthetic_data(
    n_samples: int = 10_000,
    n_features: int = 10,
    noise_std: float = 0.5,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """    
    Model: y = X @ w_true + noise
    
    Args:
        n_samples: Number of data points (N)
        n_features: Dimension of features (d)
        noise_std: Standard deviation of Gaussian noise
        random_seed: For reproducibility
        
    Returns:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        w_true: True weights used to generate data (n_features,)
    """
    rng = np.random.default_rng(random_seed)
    
    # Generate features from standard normal
    X = rng.standard_normal((n_samples, n_features))
    
    # Generate true weights (arbitrary choice, uniform [-2, 2])
    w_true = rng.uniform(-2, 2, n_features)
    
    # Generate targets with additive Gaussian noise
    noise = rng.normal(0, noise_std, n_samples)
    y = X @ w_true + noise
    
    return X, y, w_true
