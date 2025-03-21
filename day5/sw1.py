import random
import matplotlib.pyplot as plt
import numpy as np

def random_walk(num_steps):
    """
    Simulates a 1D random walk.

    Args:
        num_steps (int): The number of steps in the random walk.

    Returns:
        list: A list of integers representing the positions at each step.
              The first element is always 0.
    """
    path = [0]  # Start at position 0
    for _ in range(num_steps):
        # Choose a random step: +1 (right) or -1 (left)
        step = random.choice([-1, 1])
        # Calculate the new position by adding the step to the previous position
        new_position = path[-1] + step
        path.append(new_position)
    return path

def visualize_random_walk(path, num_steps, probability=None, end_points=None, save_path=None):
    """
    Visualizes the random walk and optionally displays the probability.
    Also displays a histogram of end points if provided.

    Args:
        path (list): The path of the random walk (list of positions).
        num_steps (int): The number of steps in the random walk.
        probability (float, optional): The probability of ending up within 3 steps of 0.
            If provided, it will be displayed on the plot.
        end_points (list, optional): A list of end points from multiple simulations.
            If provided, a histogram of these end points will be displayed.
        save_path (str, optional): Path where the plot should be saved.
            If not provided, the plot is displayed but not saved.
    """
    plt.figure(figsize=(14, 6))  # Adjust figure size for better visualization
    plt.subplot(1, 2, 1) # Create two subplots side by side

    plt.plot(path, marker='o', linestyle='-', color='blue', markersize=8)  # Plot the path
    plt.title(f'1D Random Walk ({num_steps} Steps)')  # Set the title of the plot
    plt.xlabel('Step Number')  # Label the x-axis
    plt.ylabel('Position')  # Label the y-axis
    plt.grid(True)  # Add grid lines for better readability

    # Highlight the starting and ending points with different colors
    plt.plot(0, path[0], marker='o', color='green', markersize=12, label='Start')
    plt.plot(num_steps, path[-1], marker='o', color='red', markersize=12, label='End')

    # Add a horizontal band to indicate the target region (within 3 steps of 0)
    plt.axhspan(-3, 3, alpha=0.2, color='gray', label='Target Region (-3 to +3)')

    # Display the probability if provided
    if probability is not None:
        plt.text(num_steps * 0.7, 4, f'Probability: {probability:.4f}', fontsize=12, color='black')  # Position the text

    plt.legend()  # Show the legend to identify the start, end, and target region
    plt.tight_layout()  # Adjust layout to prevent elements from overlapping

    if end_points:
        plt.subplot(1, 2, 2)  # Select the second subplot for the histogram
        plt.hist(end_points, bins=20, color='purple', alpha=0.7) # Plot the histogram
        plt.title('Histogram of End Positions')
        plt.xlabel('End Position')
        plt.ylabel('Frequency')
        plt.grid(True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()  # Display the plot

def calculate_probability(num_steps, num_simulations):
    """
    Calculates the probability of ending up within 3 steps of 0 after a random walk.

    Args:
        num_steps (int): The number of steps in the random walk.
        num_simulations (int): The number of random walks to simulate.

    Returns:
        float: The estimated probability.
    """
    end_points = []  # List to store the ending positions of all simulations
    for _ in range(num_simulations):
        path = random_walk(num_steps)  # Simulate a random walk
        end_points.append(path[-1])  # Store the final position

    # Count how many times the end point is within the target range (-3 to +3)
    favorable_outcomes = sum(-3 <= end_position <= 3 for end_position in end_points)
    # Calculate the probability
    probability = favorable_outcomes / num_simulations if num_simulations > 0 else 0
    return probability, end_points # Return both probability and end_points

if __name__ == "__main__":
    num_steps = 5  # Number of steps in the random walk
    num_simulations = 10000  # Number of simulations to run for probability estimation

    # Simulate one random walk and visualize it
    path = random_walk(num_steps)
    probability, end_points = calculate_probability(num_steps, num_simulations) # Get both
    
    # Define the filename for the saved plot
    save_filename = f"random_walk_{num_steps}steps_{num_simulations}sims.png"
    
    # Visualize and save the plot
    visualize_random_walk(path, num_steps, probability, end_points, save_path=save_filename)

    # Print the probability
    print(f"The probability of ending up within 3 steps of 0 after {num_steps} steps is approximately {probability:.4f}")
