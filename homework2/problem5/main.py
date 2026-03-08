# main.py
# This script runs the main experiments for Problem 5, comparing different aggregation rules
# under various attack scenarios. It includes plotting results and summarizing findings.


import numpy as np
import matplotlib.pyplot as plt
from data import load_mnist, partition_data
from models import LogisticRegression
from strategies import NoCompression, TopKSparsification, SignSGD, QSGD
from client import Client
from server import FederatedServer


def run_federated_experiment(strategy_name, compression, rounds=50, 
                            local_epochs=5, local_sgd_steps=None):
    """Run experiment with optional Local SGD configuration."""
    print(f"\nRunning {strategy_name}...")
    
    # Load data (same as before)
    (x_train, y_train), (x_test, y_test) = load_mnist()
    client_data = partition_data(x_train, y_train, num_clients=20, iid=True)
    
    global_model = LogisticRegression(input_dim=784, num_classes=10)
    server = FederatedServer(global_model, compression)
    
    clients = []
    for i, data in enumerate(client_data):
        client_model = LogisticRegression(input_dim=784, num_classes=10)
        clients.append(Client(i, data, client_model, compression))
    
    history = {
        'test_acc': [],
        'train_loss': [],
        'bits': [],
        'rounds': []
    }
    
    for r in range(rounds):
        selected = np.random.choice(len(clients), size=10, replace=False)
        selected_clients = [clients[i] for i in selected]
        
        server.distribute(selected_clients)
        
        client_updates = []
        for client in selected_clients:
            # Use local_sgd_steps if specified, otherwise use epochs
            if local_sgd_steps is not None:
                update = client.local_train(epochs=None, lr=0.01, 
                                          local_steps=local_sgd_steps)
            else:
                update = client.local_train(epochs=local_epochs, lr=0.01)
            client_updates.append(update)
        
        server.aggregate(client_updates)
        
        # Evaluate (same as before)
        acc = global_model.accuracy(x_test, y_test)
        loss = global_model.loss(x_train[:10000], y_train[:10000])
        
        history['test_acc'].append(acc)
        history['train_loss'].append(loss)
        history['bits'].append(server.total_bits_sent)
        history['rounds'].append(r + 1)
        
        if r % 10 == 0:
            comm_type = f"{local_sgd_steps} local steps" if local_sgd_steps else f"{local_epochs} epochs"
            print(f"  Round {r} ({comm_type}): Acc={acc:.3f}, Bits={server.total_bits_sent/1e6:.1f}M")
    
    return history


def main():
    """Compare all strategies including Local SGD variants."""
    
    # Standard strategies
    strategies = {
        'FedAvg (no compression)': (NoCompression(), None),
        'Top-10% sparsification': (TopKSparsification(k_percent=10), None),
        'SignSGD (1-bit)': (SignSGD(), None),
        'QSGD (8-bit)': (QSGD(num_levels=256), None),
        # Local SGD variants: more local computation, less frequent communication
        'Local SGD (50 steps)': (NoCompression(), 50),  # 50 steps per round
        'Local SGD (100 steps)': (NoCompression(), 100),  # 100 steps per round
    }
    
    results = {}
    for name, (compression, local_steps) in strategies.items():
        results[name] = run_federated_experiment(
            name, compression, rounds=50, 
            local_epochs=5, local_sgd_steps=local_steps
        )
    
    # Plotting (same 3 plots as before, but with Local SGD lines)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Accuracy vs Rounds
    ax = axes[0]
    for name, hist in results.items():
        linestyle = '--' if 'Local SGD' in name else '-'
        ax.plot(hist['rounds'], hist['test_acc'], label=name, 
                marker='o', markersize=3, linestyle=linestyle)
    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy vs Rounds')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss vs Rounds
    ax = axes[1]
    for name, hist in results.items():
        linestyle = '--' if 'Local SGD' in name else '-'
        ax.plot(hist['rounds'], hist['train_loss'], label=name, 
                marker='o', markersize=3, linestyle=linestyle)
    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss vs Rounds')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy vs Bits (key comparison)
    ax = axes[2]
    for name, hist in results.items():
        linestyle = '--' if 'Local SGD' in name else '-'
        ax.plot(np.array(hist['bits'])/1e6, hist['test_acc'], label=name, 
                marker='o', markersize=3, linestyle=linestyle)
    ax.set_xlabel('Total Transmitted Bits (millions)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Accuracy vs Communication Budget')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('federated_comparison_with_local_sgd.png', dpi=150)
    print("\nSaved plot to federated_comparison_with_local_sgd.png")
    plt.show()
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY FOR PART (C)")
    print("="*70)
    print(f"{'Method':<30} {'Final Acc':>10} {'Bits (M)':>12} {'Rounds':>10}")
    print("-"*70)
    for name, hist in results.items():
        final_acc = hist['test_acc'][-1]
        total_bits = hist['bits'][-1] / 1e6
        rounds = len(hist['rounds'])
        print(f"{name:<30} {final_acc:>10.3f} {total_bits:>12.1f} {rounds:>10}")


if __name__ == "__main__":
    main()
