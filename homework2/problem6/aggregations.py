# aggregations.py
# This script implements various robust aggregation rules for federated learning. These rules are designed
# to mitigate the impact of Byzantine clients that may send malicious updates. The implemented rules include
# coordinate-wise median, trimmed mean, and Krum, in addition to the standard mean aggregation for comparison.


import numpy as np
from scipy.spatial.distance import cdist


def mean_aggregation(updates, weights=None):
    """
    Standard FedAvg: weighted mean of updates.
    Vulnerable to Byzantine attacks.
    """
    updates = np.array(updates)
    if weights is None:
        weights = np.ones(len(updates)) / len(updates)
    else:
        weights = np.array(weights) / np.sum(weights)
    
    return np.average(updates, axis=0, weights=weights)


def coordinate_wise_median(updates, weights=None):
    """
    Coordinate-wise median: robust to outliers in each dimension.
    Takes median of each parameter across all clients.
    """
    updates = np.array(updates)
    return np.median(updates, axis=0)


def trimmed_mean(updates, weights=None, trim_ratio=0.3):
    """
    Trimmed mean: remove extreme values and average the rest.
    Removes top and bottom trim_ratio fraction before averaging.
    """
    updates = np.array(updates)
    n = len(updates)
    trim_count = int(n * trim_ratio)
    
    # Sort each coordinate and trim
    sorted_updates = np.sort(updates, axis=0)
    trimmed = sorted_updates[trim_count:n-trim_count]
    
    return np.mean(trimmed, axis=0)


def krum_aggregation(updates, weights=None, num_byzantine=None):
    """
    Krum: select the update closest to its neighbors.
    Robust when num_byzantine < n/2.
    """
    updates = np.array(updates)
    n = len(updates)
    
    if num_byzantine is None:
        num_byzantine = n // 2 - 1  # Conservative estimate
    
    # Compute pairwise distances
    distances = cdist(updates, updates, metric='euclidean')
    
    # For each update, compute sum of distances to nearest n - num_byzantine - 2 neighbors
    scores = []
    for i in range(n):
        # Sort distances to all other updates
        sorted_dists = np.sort(distances[i])
        # Sum distances to closest n - num_byzantine - 2 neighbors (excluding self)
        score = np.sum(sorted_dists[1:n - num_byzantine])
        scores.append(score)
    
    # Select update with minimum score
    selected_idx = np.argmin(scores)
    return updates[selected_idx]
