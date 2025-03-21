import matplotlib.pyplot as plt
import random
import numpy as np

def simulate_financial_trajectory(time_steps=500):
    """
    Simulate a financial trajectory (random walk) and track capital over time.
    
    Args:
        time_steps: Number of time steps in the simulation
        
    Returns:
        capital: The entire capital trajectory
        final_capital: The final capital value
        non_negative_time_fraction: Fraction of time spent with non-negative capital
    """
    capital = [0]  # Initial capital is 0
    
    for _ in range(time_steps):
        # Each step can increase or decrease capital by 1 unit with equal probability
        capital.append(capital[-1] + random.choice([-1, 1]))
    
    # Calculate fraction of time spent with non-negative capital
    non_negative_time_fraction = sum(1 for c in capital if c >= 0) / len(capital)
    
    return capital, capital[-1], non_negative_time_fraction

def analyze_financial_trajectories():
    # Parameters
    num_trajectories = 10000  # Large number of trajectories for accurate approximation
    time_steps = 500  # As requested in the problem
    threshold = 0.75  # Three-quarters of the time
    
    # Simulate trajectories
    print(f"Simulating {num_trajectories:,} financial trajectories with {time_steps} time steps each...")
    results = [simulate_financial_trajectory(time_steps) for _ in range(num_trajectories)]
    
    # Extract results
    all_trajectories = [result[0] for result in results]
    final_capitals = [result[1] for result in results]
    non_negative_fractions = [result[2] for result in results]
    
    # Calculate the probability of having non-negative capital at least 75% of the time
    favorable_outcomes = sum(1 for fraction in non_negative_fractions if fraction >= threshold)
    probability = favorable_outcomes / num_trajectories
    
    print(f"\nResults from Financial Trajectory Analysis:")
    print(f"Probability of having non-negative capital at least {threshold*100}% of the time: {probability:.4f}")
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # 1. Plot all trajectories with transparency
    # For better visibility, let's plot a subset (e.g., 100 random ones)
    sample_size = min(100, num_trajectories)
    random_indices = random.sample(range(num_trajectories), sample_size)
    
    for idx in random_indices:
        trajectory = all_trajectories[idx]
        # Plot with high transparency
        ax1.plot(trajectory, linewidth=0.5, alpha=0.1, color='blue')
    
    # Highlight a few interesting trajectories
    # Find trajectories that end with highest and lowest values
    max_idx = final_capitals.index(max(final_capitals))
    min_idx = final_capitals.index(min(final_capitals))
    
    # Plot these specific trajectories with lower transparency
    ax1.plot(all_trajectories[max_idx], color='green', linewidth=1.5, alpha=0.8, label='Highest final')
    ax1.plot(all_trajectories[min_idx], color='red', linewidth=1.5, alpha=0.8, label='Lowest final')
    
    # Find a trajectory close to average final capital
    avg_capital = sum(final_capitals) / len(final_capitals)
    avg_idx = min(range(len(final_capitals)), key=lambda i: abs(final_capitals[i] - avg_capital))
    ax1.plot(all_trajectories[avg_idx], color='purple', linewidth=1.5, alpha=0.8, label='Close to average')
    
    ax1.set_title(f'Sample of {sample_size} Financial Trajectories ({time_steps} Steps)', fontsize=14)
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Capital', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.6)
    ax1.legend()
    
    # 2. Distribution of final capitals
    counts, bins, patches = ax2.hist(
        final_capitals, 
        bins=50, 
        density=True,
        alpha=0.7, 
        color='green', 
        edgecolor='black'
    )
    
    # Fit a normal distribution
    mean = np.mean(final_capitals)
    std = np.std(final_capitals)
    x = np.linspace(min(final_capitals), max(final_capitals), 100)
    p = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    ax2.plot(x, p, 'r--', linewidth=2, label=f'Normal Fit: μ={mean:.2f}, σ={std:.2f}')
    
    ax2.set_title(f'Distribution of Final Capital ({num_trajectories:,} trajectories)', fontsize=14)
    ax2.set_xlabel('Final Capital', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    # 3. Log-scale visualization of final capital distribution
    ax3.hist(
        final_capitals, 
        bins=50, 
        density=True,
        alpha=0.7, 
        color='purple', 
        edgecolor='black'
    )
    ax3.set_title('Log-scale Distribution of Final Capital', fontsize=14)
    ax3.set_xlabel('Final Capital', fontsize=12)
    ax3.set_ylabel('Log Probability Density', fontsize=12)
    ax3.set_yscale('log')  # Set y-axis to logarithmic scale
    ax3.grid(alpha=0.3)
    
    # Add normal curve to log plot
    ax3.plot(x, p, 'r--', linewidth=2, label=f'Normal Fit: μ={mean:.2f}, σ={std:.2f}')
    ax3.legend()
    
    # 4. Distribution of time fraction with non-negative capital
    ax4.hist(
        non_negative_fractions, 
        bins=30, 
        density=True,
        alpha=0.7, 
        color='orange', 
        edgecolor='black'
    )
    ax4.set_title('Fraction of Time with Non-Negative Capital', fontsize=14)
    ax4.set_xlabel('Fraction of Time with Capital ≥ 0', fontsize=12)
    ax4.set_ylabel('Probability Density', fontsize=12)
    ax4.grid(alpha=0.3)
    
    # Add key metrics to the plot
    ax4.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
    ax4.axvline(x=np.mean(non_negative_fractions), color='green', linestyle='-', 
                label=f'Mean: {np.mean(non_negative_fractions):.3f}')
    ax4.text(0.05, 0.95, f'P(Non-negative time ≥ {threshold}) = {probability:.4f}', 
             transform=ax4.transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('financial_trajectory_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    random.seed(42)  # For reproducibility
    analyze_financial_trajectories()