import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the expected value of a function g(x) with respect to the probability density function p(x)
def expected_value(g, p, a, b, samples=1000000):
    xs = np.random.uniform(a, b, samples)
    gxs = g(xs)
    pxs = p(xs)
    return np.mean(gxs * pxs)

# Given probability density function
def pdf(x):
    return 2 * x

# Expected value functions for x and x^2
def g_x(x):
    return x

def g_x2(x):
    return x**2

# Calculate expected values
exp_x = expected_value(g_x, pdf, 0, 1)
exp_x2 = expected_value(g_x2, pdf, 0, 1)

# Variance and standard deviation
var_x = exp_x2 - exp_x**2
std_x = np.sqrt(var_x)

# Monte Carlo simulation to estimate the distribution of means
def monte_carlo_simulation(n_batches, samples_per_batch, pdf):
    batch_means = []
    for _ in range(n_batches):
        samples = np.random.uniform(0, 1, samples_per_batch)
        weights = pdf(samples)
        batch_means.append(np.mean(samples * weights))
    return np.array(batch_means)

# Parameters for Monte Carlo simulation
n_batches = 10000
samples_per_batch = 1250

# Running the Monte Carlo simulation
batch_means = monte_carlo_simulation(n_batches, samples_per_batch, pdf)

# Empirical rule check
mean_of_means = np.mean(batch_means)
std_of_mean = np.std(batch_means, ddof=1)
within_1_std = np.mean((batch_means >= mean_of_means - std_of_mean) & (batch_means <= mean_of_means + std_of_mean))
within_2_std = np.mean((batch_means >= mean_of_means - 2 * std_of_mean) & (batch_means <= mean_of_means + 2 * std_of_mean))
within_3_std = np.mean((batch_means >= mean_of_means - 3 * std_of_mean) & (batch_means <= mean_of_means + 3 * std_of_mean))

# Plotting the distribution of means
plt.figure(figsize=(10, 5))
plt.hist(batch_means, bins=50, alpha=0.5, label='Batch Means')
plt.axvline(mean_of_means, color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(mean_of_means + std_of_mean, color='green', linestyle='dashed', linewidth=2, label='+1 Std Dev')
plt.axvline(mean_of_means - std_of_mean, color='green', linestyle='dashed', linewidth=2, label='-1 Std Dev')
plt.axvline(mean_of_means + 2 * std_of_mean, color='blue', linestyle='dashed', linewidth=2, label='+2 Std Dev')
plt.axvline(mean_of_means - 2 * std_of_mean, color='blue', linestyle='dashed', linewidth=2)
plt.axvline(mean_of_means + 3 * std_of_mean, color='yellow', linestyle='dashed', linewidth=2, label='+3 Std Dev')
plt.axvline(mean_of_means - 3 * std_of_mean, color='yellow', linestyle='dashed', linewidth=2)
plt.xlabel('Mean of Batches')
plt.ylabel('Frequency')
plt.legend()
plt.title('Monte Carlo Simulation of Batch Means')
plt.grid(True)
plt.show()

# Output the results
print(f"Expected <x>: {exp_x:.4f}, Expected <x^2>: {exp_x2:.4f}")
print(f"Variance σ^2: {var_x:.4f}, Std Dev σ: {std_x:.4f}")
print(f"Mean of batch means: {mean_of_means:.4f}")
print(f"Std Dev of batch means: {std_of_mean:.4f}")
print(f"Fraction within ±1σ: {within_1_std:.4f}")
print(f"Fraction within ±2σ: {within_2_std:.4f}")
print(f"Fraction within ±3σ: {within_3_std:.4f}")