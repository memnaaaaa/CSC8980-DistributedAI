# data.py
# This cript handles loading the MNIST dataset and partitioning it among clients for federated learning.


import numpy as np
from tensorflow import keras


def load_mnist():
    """Load and preprocess MNIST."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize and flatten
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    # Binary classification: digit 0 vs 1 (simpler for demo)
    # Or use all 10 classes with one-hot encoding
    train_mask = (y_train < 10)  # Keep all digits
    test_mask = (y_test < 10)
    
    x_train, y_train = x_train[train_mask], y_train[train_mask]
    x_test, y_test = x_test[test_mask], y_test[test_mask]
    
    return (x_train, y_train), (x_test, y_test)


def partition_data(x, y, num_clients=20, iid=True):
    """
    Partition data among clients.
    
    Args:
        iid: If True, random shuffle. If False, sort by label (non-iid).
    """
    if iid:
        # Random shuffle
        indices = np.random.permutation(len(x))
        x, y = x[indices], y[indices]
        # Split evenly
        client_data = []
        samples_per_client = len(x) // num_clients
        for i in range(num_clients):
            start = i * samples_per_client
            end = (i + 1) * samples_per_client if i < num_clients - 1 else len(x)
            client_data.append((x[start:end], y[start:end]))
    else:
        # Non-IID: sort by label, then partition
        sorted_indices = np.argsort(y)
        x, y = x[sorted_indices], y[sorted_indices]
        client_data = []
        samples_per_client = len(x) // num_clients
        for i in range(num_clients):
            start = i * samples_per_client
            end = (i + 1) * samples_per_client if i < num_clients - 1 else len(x)
            client_data.append((x[start:end], y[start:end]))
    
    return client_data
