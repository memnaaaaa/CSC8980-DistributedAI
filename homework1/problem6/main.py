# homework1/problem6/main.py
# This script compares Full Gradient Descent, Mini-batch SGD, and Pure SGD


# Importing necessary libraries
import numpy as np # for numerical computations
import matplotlib.pyplot as plt # for plotting
import sys # for path manipulations
sys.path.append('../problem5') # to access data_generator from problem5
from data_generator import generate_synthetic_data # data generation
from optimizer import GradientDescent, MiniBatchSGD, PureSGD # optimizers


# Function to compute closed-form solution
def closed_form(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def main():
    # Generate data (same as Problem 5)
    X, y, w_true = generate_synthetic_data(n_samples=10000, n_features=10)
    w_star = closed_form(X, y)
    print(f"Optimal loss: {np.mean((X @ w_star - y)**2):.4f}\n")
    
    # Run comparisons
    results = {}
    
    # 1. Full GD
    print("Running Full GD...")
    gd = GradientDescent(step_size=0.4, max_iter=200)
    _, gd_loss = gd.fit(X, y)
    results['GD'] = {
        'loss': gd_loss,
        'time_per_iter': np.mean(gd.time_history) * 1000,  # ms
        'total_time': sum(gd.time_history)
    }
    print(f"  Per-iteration: {results['GD']['time_per_iter']:.3f}ms, Final loss: {gd_loss[-1]:.4f}")
    
    # 2. Mini-batch SGD
    print("Running Mini-batch SGD (batch=32)...")
    mb = MiniBatchSGD(step_size=0.05, max_iter=200, batch_size=32)
    _, mb_loss = mb.fit(X, y)
    results['MiniBatch'] = {
        'loss': mb_loss,
        'time_per_iter': np.mean(mb.time_history) * 1000,
        'total_time': sum(mb.time_history)
    }
    print(f"  Per-iteration: {results['MiniBatch']['time_per_iter']:.3f}ms, Final loss: {mb_loss[-1]:.4f}")
    
    # 3. Pure SGD
    print("Running Pure SGD...")
    sgd = PureSGD(step_size=0.005, max_iter=2000)
    _, sgd_loss = sgd.fit(X, y)
    results['SGD'] = {
        'loss': sgd_loss,
        'time_per_iter': np.mean(sgd.time_history) * 1000,
        'total_time': sum(sgd.time_history)
    }
    print(f"  Per-iteration: {results['SGD']['time_per_iter']:.3f}ms, Final loss: {sgd_loss[-1]:.4f}")
    
    # Summary table
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Method':<12} {'ms/iter':>10} {'Total(s)':>10} {'Final Loss':>12}")
    print("-"*60)
    for name, r in results.items():
        print(f"{name:<12} {r['time_per_iter']:>10.3f} {r['total_time']:>10.3f} {r['loss'][-1]:>12.4f}")
    
    # Plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(results['GD']['loss'], label='GD')
    plt.plot(results['MiniBatch']['loss'], label='Mini-batch')
    plt.plot(results['SGD']['loss'], label='SGD')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Convergence (Loss vs Iteration)')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    # Plot against wall-clock time
    gd_time = np.cumsum(gd.time_history)
    mb_time = np.cumsum(mb.time_history)
    sgd_time = np.cumsum(sgd.time_history)
    plt.plot(gd_time, results['GD']['loss'], label='GD')
    plt.plot(mb_time, results['MiniBatch']['loss'], label='Mini-batch')
    plt.plot(sgd_time[::10], results['SGD']['loss'], label='SGD')  # SGD recorded every 10
    plt.xlabel('Time (seconds)')
    plt.ylabel('Loss')
    plt.title('Convergence (Loss vs Time)')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    names = list(results.keys())
    times = [results[n]['time_per_iter'] for n in names]
    plt.bar(names, times)
    plt.ylabel('Time per iteration (ms)')
    plt.title('Per-Iteration Cost')
    
    plt.tight_layout()
    plt.savefig('comparison.png')
    print("\nSaved plot to comparison.png")


if __name__ == "__main__":
    main()