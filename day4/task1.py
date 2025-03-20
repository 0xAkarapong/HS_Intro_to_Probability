import matplotlib.pyplot as plt
import random
import numpy as np

# Random walk

def single_random_walk(steps=1000):
    """Simulate a single random walk and return the final position"""
    position = 0
    for _ in range(steps):
        position += random.choice([-1, 1])
    return position

def plot():
    # Parameters
    num_walks = 10000
    step_count = 1000
    
    # Simulate 10,000 random walks and track final positions
    final_positions = [single_random_walk(step_count) for _ in range(num_walks)]
    
    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot a sample random walk path (just one example)
    steps = [0]
    for i in range(1, step_count+1):
        steps.append(steps[i-1] + random.choice([-1, 1]))
    
    ax1.plot(steps, color='blue', linewidth=1.5)
    ax1.set_title('Sample Random Walk Path', fontsize=14)
    ax1.set_xlabel('Step Number', fontsize=12)
    ax1.set_ylabel('Position', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.plot(0, steps[0], 'go', markersize=8, label='Start')
    ax1.plot(len(steps)-1, steps[-1], 'ro', markersize=8, label='End')
    ax1.legend()
    
    # Plot histogram of final positions from 10,000 walks
    counts, bins, patches = ax2.hist(
        final_positions, 
        bins=50, 
        density=True,  # Normalize to create a probability density
        alpha=0.7, 
        color='green', 
        edgecolor='black'
    )
    
    # Fit a normal distribution
    mean = np.mean(final_positions)
    std = np.std(final_positions)
    x = np.linspace(min(final_positions), max(final_positions), 100)
    p = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    ax2.plot(x, p, 'r--', linewidth=2, label=f'Normal Fit: μ={mean:.2f}, σ={std:.2f}')
    
    ax2.set_title('Histogram of Final Positions (10,000 walks)', fontsize=14)
    ax2.set_xlabel('Final Position', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    # Plot logarithmic histogram of final positions
    ax3.hist(
        final_positions, 
        bins=50, 
        density=True,
        alpha=0.7, 
        color='purple', 
        edgecolor='black'
    )
    ax3.set_title('Log-scale Histogram of Final Positions', fontsize=14)
    ax3.set_xlabel('Final Position', fontsize=12)
    ax3.set_ylabel('Log Probability Density', fontsize=12)
    ax3.set_yscale('log')  # Set y-axis to logarithmic scale
    ax3.grid(alpha=0.3)
    
    # Add normal distribution curve to log plot as well
    ax3.plot(x, p, 'r--', linewidth=2, label=f'Normal Fit: μ={mean:.2f}, σ={std:.2f}')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('random_walk_with_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    random.seed(42)  # For reproducibility
    plot()