# server.py
# This script defines the RobustFederatedServer class, which orchestrates the federated learning process
# using robust aggregation rules. The server aggregates updates from clients, applies the chosen
# robust aggregation rule, and updates the global model accordingly. The server can handle different
# numbers of Byzantine clients and supports multiple aggregation strategies for comparison in experiments.


import numpy as np
from aggregations import mean_aggregation, coordinate_wise_median, trimmed_mean, krum_aggregation


class RobustFederatedServer:
    """Central server with robust aggregation rules."""
    
    AGGREGATION_RULES = {
        'mean': mean_aggregation,
        'median': coordinate_wise_median,
        'trimmed_mean': trimmed_mean,
        'krum': krum_aggregation,
    }
    
    def __init__(self, model, aggregation_rule='mean', num_byzantine=0):
        self.global_model = model
        self.aggregation_rule = aggregation_rule
        self.aggregator = self.AGGREGATION_RULES[aggregation_rule]
        self.num_byzantine = num_byzantine
        self.round = 0
        
    def aggregate(self, client_updates, client_samples):
        """
        Aggregate client updates using selected robust rule.
        
        Args:
            client_updates: list of parameter updates from clients
            client_samples: list of sample counts (for weighted aggregation)
        """
        # Normalize weights by sample count
        total_samples = sum(client_samples)
        weights = [n / total_samples for n in client_samples]
        
        # Apply aggregation rule
        if self.aggregation_rule == 'krum':
            aggregated = self.aggregator(client_updates, num_byzantine=self.num_byzantine)
        elif self.aggregation_rule == 'trimmed_mean':
            aggregated = self.aggregator(client_updates, weights)
        else:
            aggregated = self.aggregator(client_updates, weights)
        
        # Apply to global model
        current_params = self.global_model.get_params()
        new_params = current_params + aggregated
        self.global_model.set_params(new_params)
        self.round += 1
        
    def get_model_params(self):
        return self.global_model.get_params()
    
    def distribute(self, clients):
        """Send global model to all clients."""
        params = self.get_model_params()
        for client in clients:
            client.set_model(params)
