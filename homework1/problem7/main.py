# homework1/problem7/main.py
# This script analyzes gradient descent on a 1D non-convex function.


# Importing necessary libraries
import numpy as np # for numerical computations
import matplotlib.pyplot as plt # for plotting
import sys # for path manipulations
from pathlib import Path 
# Adjusting path to import local modules
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
from optimizer import f, grad_f, gradient_descent # GD implementation


def analyze_function():
    """Part (a): Plot function and identify critical points."""
    x = np.linspace(-1, 3, 400)
    y = f(x)
    
    # Critical points
    x_min = 2.25  # Local minimum
    f_min = f(x_min)
    
    plt.figure(figsize=(10, 6))
    
    # Plot 1: Function landscape
    plt.subplot(2, 2, 1)
    plt.plot(x, y, 'b-', linewidth=2, label='f(x)')
    plt.axvline(x=x_min, color='red', linestyle='--', alpha=0.5, label=f'Local min ({x_min})')
    plt.axvline(x=0, color='orange', linestyle='--', alpha=0.5, label='Saddle (x=0)')
    plt.scatter([0, 2.25], [f(0), f(2.25)], color=['orange', 'red'], s=100, zorder=5)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('(a) Function Landscape')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-1, 3)
    
    return x_min, f_min


def run_gradient_descent():
    """Parts (b) & (c): Run GD from three different starting points."""
    
    # Starting points as specified
    starting_points = [0, 1.5, 2.5]
    step_size = 0.05  # Stable: max step < 2/Lipschitz near x=2.5 (~2/30 ≈ 0.066)
    max_iter = 50
    
    results = {}
    
    for x0 in starting_points:
        traj, vals = gradient_descent(x0, step_size, max_iter)
        results[x0] = {
            'trajectory': traj,
            'values': vals,
            'final_x': traj[-1],
            'final_f': vals[-1]
        }
    
    return results, step_size


def plot_convergence(results, true_min_x, true_min_f):
    """Part (c): Plot convergence behaviors."""
    
    # Plot 2: Trajectory of x_k
    plt.subplot(2, 2, 2)
    for x0, data in results.items():
        plt.plot(data['trajectory'], marker='o', markersize=3, 
                label=f'x0={x0}', alpha=0.8)
    plt.axhline(y=true_min_x, color='red', linestyle='--', 
                alpha=0.5, label=f'True min (x={true_min_x})')
    plt.xlabel('Iteration')
    plt.ylabel('x_k')
    plt.title('(c) Convergence: x_k vs Iteration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Function values f(x_k)
    plt.subplot(2, 2, 3)
    for x0, data in results.items():
        plt.plot(data['values'], marker='o', markersize=3, 
                label=f'x0={x0}', alpha=0.8)
    plt.axhline(y=true_min_f, color='red', linestyle='--', 
                alpha=0.5, label=f'Min value ({true_min_f:.2f})')
    plt.xlabel('Iteration')
    plt.ylabel('f(x_k)')
    plt.title('(c) Convergence: f(x_k) vs Iteration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Phase space (x vs f(x))
    plt.subplot(2, 2, 4)
    x_range = np.linspace(-1, 3, 100)
    plt.plot(x_range, f(x_range), 'k-', alpha=0.3, label='f(x)')
    colors = ['orange', 'green', 'blue']
    for i, (x0, data) in enumerate(results.items()):
        plt.plot(data['trajectory'], data['values'], 'o-', 
                color=colors[i], markersize=3, label=f'x0={x0}')
    plt.scatter([2.25], [f(2.25)], color='red', s=200, marker='*', 
               zorder=5, label='Global minimum')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Optimization Trajectories')
    plt.legend()
    plt.grid(True, alpha=0.3)


def print_analysis(results, step_size):
    """Genuine observations for the report."""
    print("\n" + "="*60)
    print("PROBLEM 7 ANALYSIS")
    print("="*60)
    print(f"Step size used: {step_size}")
    print(f"True local minimum: x = 2.25, f(x) = {f(2.25):.4f}")
    print(f"Saddle point at x = 0: f'(0) = {grad_f(0)}, f(0) = {f(0)}")
    print("-"*60)
    
    for x0, data in results.items():
        print(f"\nStarting from x0 = {x0}:")
        print(f"  Final x: {data['final_x']:.4f}")
        print(f"  Final f(x): {data['final_f']:.4f}")
        
        if x0 == 0:
            print("  → STUCK at saddle point! Gradient is zero, so no movement.")
        elif abs(data['final_x'] - 2.25) < 0.01:
            print("  → CONVERGED to local minimum correctly.")
        else:
            print("  → Did not converge (check step size or iterations).")


def main():
    # Part (a)
    true_min_x, true_min_f = analyze_function()
    
    # Parts (b) and (c)
    results, step_size = run_gradient_descent()
    plot_convergence(results, true_min_x, true_min_f)
    
    plt.tight_layout()
    plt.savefig('problem7_convergence.png', dpi=150, bbox_inches='tight')
    print("Saved figure to problem7_convergence.png")
    plt.show()
    
    print_analysis(results, step_size)


if __name__ == "__main__":
    main()