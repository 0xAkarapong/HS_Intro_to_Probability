import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Generate a uniform dataset over [0,1]
n = 100000  # Number of samples
alpha = np.random.uniform(0, 1, n)

# Compute the transformations
alpha_squared = alpha**2
alpha_sqrt = np.sqrt(alpha)
# Need to handle potential division by zero for 1/sqrt(alpha)
epsilon = 1e-10  # Small threshold to avoid division issues
alpha_filtered = alpha[alpha > epsilon]
alpha_inv_sqrt = 1/np.sqrt(alpha_filtered)

# Create plots
plt.figure(figsize=(15, 10))

# Plot for alpha (original uniform distribution)
plt.subplot(2, 2, 1)
plt.hist(alpha, bins=50, alpha=0.7, color='blue', density=True)
plt.title('Distribution of $\\alpha$ ~ Uniform[0,1]')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(alpha=0.3)

# Plot for alpha^2
plt.subplot(2, 2, 2)
plt.hist(alpha_squared, bins=50, alpha=0.7, color='red', density=True)
plt.title('Distribution of $\\alpha^2$')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(alpha=0.3)

# Plot for sqrt(alpha)
plt.subplot(2, 2, 3)
plt.hist(alpha_sqrt, bins=50, alpha=0.7, color='green', density=True)
plt.title('Distribution of $\\sqrt{\\alpha}$')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(alpha=0.3)

# Plot for 1/sqrt(alpha)
plt.subplot(2, 2, 4)
plt.hist(alpha_inv_sqrt, bins=50, alpha=0.7, color='purple', density=True)
plt.title('Distribution of $1/\\sqrt{\\alpha}$')
plt.xlabel('Value')
plt.ylabel('Density')
plt.xlim(0, 10)  # Clip x-axis to see the main part of the distribution
plt.text(0.5, 0.9, 'Note: X-axis clipped at 10 for visibility', 
         transform=plt.gca().transAxes, fontsize=8)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()