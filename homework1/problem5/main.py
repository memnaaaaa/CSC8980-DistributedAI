# main.py
# This script is the main execution point for Problem 5.
# Generates data, runs GD experiments, and visualizes convergence.

import sys
from pathlib import Path

# Adjusting path to import local modules
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# Importing necessary libraries
import numpy as np # for numerical computations
import matplotlib.pyplot as plt # for plotting
from data_generator import generate_synthetic_data # for data generation
from optimizer import LinearRegressionGD # GD optimizer implementation

# Function to compute closed-form solution
def compute_closed_form_solution(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute w* = (X^T X)^{-1} X^T y using pseudo-inverse for stability.
    
    This is the optimal solution that GD should converge to.
    """
    return np.linalg.pinv(X.T @ X) @ X.T @ y

# Function to run experiments with different step sizes
def analyze_step_sizes(X: np.ndarray, y: np.ndarray, w_star: np.ndarray):
    """
    Part (d): Compare different step sizes.
    
    We compute the Lipschitz constant L = 2*lambda_max(X^T X)/N to 
    determine theoretical stability limits.
    """
    n_samples = X.shape[0]
    
    # Compute eigenvalues of Hessian to determine step size bounds
    # Hessian of MSE = (2/N) * X^T X
    eigenvalues = np.linalg.eigvalsh((2 / n_samples) * (X.T @ X))
    L_max = np.max(eigenvalues)  # Largest eigenvalue (Lipschitz constant)
    L_min = np.min(eigenvalues)
    condition_number = L_max / L_min
    
    print(f"Hessian eigenvalue range: [{L_min:.4f}, {L_max:.4f}]")
    print(f"Condition number κ: {condition_number:.2f}")
    print(f"Theoretical max step size for stability: {2/L_max:.4f}")
    print("-" * 50)
    
    # Define step sizes for comparison
    step_sizes = {
        "too_small": 0.001,           # Very conservative
        "reasonable": 1.0 / L_max,    # ~1/L, theoretically good
        "too_large": 2.5 / L_max      # > 2/L, should diverge/oscillate
    }
    
    results = {}
    max_iter = 200
    
    for label, alpha in step_sizes.items():
        print(f"Running GD with step size α={alpha:.4f} ({label})...")
        gd = LinearRegressionGD(step_size=alpha, max_iterations=max_iter)
        losses, errors = gd.fit(X, y, w_star=w_star)
        results[label] = {
            "losses": losses,
            "errors": errors,
            "alpha": alpha,
            "final_loss": losses[-1]
        }
    
    return results, step_sizes

# Function to plot convergence results
def plot_convergence(results: dict, w_star: np.ndarray, save_path: str = "convergence_plots.png"):
    """
    Part (c) & (d): Generate required plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Loss vs Iterations (detailed view)
    ax = axes[0, 0]
    for label, data in results.items():
        ax.plot(data["losses"], label=f"α={data['alpha']:.4f} ({label})", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Loss vs Iterations (Full View)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss vs Iterations (log scale, first 50 iterations)
    ax = axes[0, 1]
    for label, data in results.items():
        if data["losses"][-1] < 10:  # Only plot if didn't diverge too badly
            ax.plot(data["losses"][:50], label=f"α={data['alpha']:.4f} ({label})", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_yscale("log")
    ax.set_title("Loss vs Iterations (First 50 Iterations, Log Scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Parameter error ||w_k - w*||
    ax = axes[1, 0]
    for label, data in results.items():
        if len(data["errors"]) > 0:
            ax.plot(data["errors"], label=f"α={data['alpha']:.4f} ({label})", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("||w_k - w*||_2")
    ax.set_title("Parameter Error vs Iterations")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Convergence comparison summary
    ax = axes[1, 1]
    labels = list(results.keys())
    final_losses = [results[l]["final_loss"] for l in labels]
    colors = ["green" if l == "reasonable" else "orange" if l == "too_small" else "red" for l in labels]
    bars = ax.bar(labels, final_losses, color=colors, alpha=0.7)
    ax.set_ylabel("Final Loss")
    ax.set_title("Final Loss Comparison")
    ax.set_yscale("log")
    
    # Add value labels on bars
    for bar, loss in zip(bars, final_losses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlots saved to {save_path}")
    plt.show()
    
    return fig

# Main execution function
def main():
    print("=" * 60)
    print("Problem 5: Gradient Descent for Linear Regression")
    print("=" * 60)
    
    # Part (a): Generate synthetic data
    print("\n[Part (a)] Generating synthetic data...")
    X, y, w_true = generate_synthetic_data(n_samples=10_000, n_features=10, noise_std=0.5)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"True weights norm: {np.linalg.norm(w_true):.4f}")
    
    # Part (b) & (c): Compute closed-form solution for reference
    print("\n[Part (b)] Computing closed-form solution w*...")
    w_star = compute_closed_form_solution(X, y)
    print(f"Closed-form solution norm: {np.linalg.norm(w_star):.4f}")
    print(f"Distance ||w_true - w*||: {np.linalg.norm(w_true - w_star):.4f}")
    
    # Verify closed-form is indeed optimal
    opt_loss = np.mean((X @ w_star - y) ** 2)
    print(f"Optimal loss (MSE): {opt_loss:.6f}")
    
    # Part (d): Run experiments with different step sizes
    print("\n[Part (d)] Running gradient descent with different step sizes...")
    results, step_sizes = analyze_step_sizes(X, y, w_star)
    
    # Print observations
    print("\n" + "=" * 60)
    print("OBSERVATIONS:")
    print("=" * 60)
    for label, data in results.items():
        print(f"\n{label.upper()} (α={data['alpha']:.4f}):")
        print(f"  - Initial loss: {data['losses'][0]:.4f}")
        print(f"  - Final loss: {data['final_loss']:.4f}")
        if len(data["errors"]) > 0:
            print(f"  - Final ||w - w*||: {data['errors'][-1]:.6f}")
        
        if label == "too_small":
            print("  → Observation: Slow convergence, monotonic decrease")
        elif label == "reasonable":
            convergence_rate = data["errors"][10] / data["errors"][5] if len(data["errors"]) > 10 else 0
            print(f"  → Observation: Fast linear convergence (rate ~{convergence_rate:.3f})")
        elif label == "too_large":
            if data["final_loss"] > 1e6:
                print("  → Observation: DIVERGENCE - loss explodes due to overshooting")
            else:
                print("  → Observation: Oscillation or slow convergence due to zig-zagging")
    
    # Generate plots
    print("\n[Part (c)] Generating convergence plots...")
    plot_convergence(results, w_star)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
