import numpy as np
import matplotlib.pyplot as plt


# Assuming f(x) = 2x for the weighting function
def f(x):
    return 2 * x


# Function to calculate weighted sample statistics
def calculate_weighted_statistics(samples):
    weights = f(samples)
    weighted_samples = weights * samples
    mean_Wx = np.mean(weighted_samples)
    mean_Wx2 = np.mean(weighted_samples ** 2)
    return mean_Wx, mean_Wx2


# Function to perform the biased Monte Carlo simulation
def biased_monte_carlo_simulation(nb_batches, samples_per_batch):
    batch_means_Wx = []
    batch_means_Wx2 = []

    for _ in range(nb_batches):
        # Biased sampling from a uniform distribution
        samples = np.random.uniform(0, 1, samples_per_batch)
        mean_Wx, mean_Wx2 = calculate_weighted_statistics(samples)
        batch_means_Wx.append(mean_Wx)
        batch_means_Wx2.append(mean_Wx2)

    # Calculate the mean of batch means for Wx and (Wx)^2
    mean_of_means_Wx = np.mean(batch_means_Wx)
    mean_of_means_Wx2 = np.mean(batch_means_Wx2)

    # Calculate the variance and standard deviation of Wx across all batches
    variance_Wx = np.var(batch_means_Wx)
    std_Wx = np.sqrt(variance_Wx)

    return mean_of_means_Wx, mean_of_means_Wx2, std_Wx, variance_Wx, batch_means_Wx


# Parameters for the Monte Carlo simulation
nb_batches = 10000
samples_per_batch = 1250

# Run the biased simulation
mean_of_means_Wx, mean_of_means_Wx2, std_Wx, variance_Wx, batch_means_Wx = biased_monte_carlo_simulation(nb_batches,
                                                                                                         samples_per_batch)

# Expected values from the original distribution f(x)
expected_mean_Wx = 2 / 3  # Integral of x * f(x) over the interval [0, 1]
expected_variance_Wx = 1 / 18  # Integral of (x^2 * f(x)) - (expected_mean_Wx)^2 over the interval [0, 1]

# Comparing the standard deviation of the Monte Carlo results with the expected standard deviation
is_good_biasing = std_Wx < np.sqrt(expected_variance_Wx)

# Output the results and comparison
print(f"Results of the biased Monte Carlo simulation:")
print(f"Mean of Wx: {mean_of_means_Wx:.5f} (Expected: {expected_mean_Wx:.5f})")
print(f"Mean of (Wx)^2: {mean_of_means_Wx2:.5f}")
print(f"Standard Deviation of Wx: {std_Wx:.5f} (Expected: {np.sqrt(expected_variance_Wx):.5f})")
print("This biasing scheme is considered " + (
    "good" if is_good_biasing else "not good") + " based on the standard deviation comparison.")
print("")

# Check if Wx falls within the expected bounds
z_scores = [1, 2, 3]
for z in z_scores:
    within_z_std = (mean_of_means_Wx + z * std_Wx > expected_mean_Wx) and \
                   (mean_of_means_Wx - z * std_Wx < expected_mean_Wx)
    print(f"The mean of Wx falls within {z} standard deviation(s): {'Yes' if within_z_std else 'No'}")

# Plotting the results for visualization
plt.figure(figsize=(12, 7))
plt.hist(batch_means_Wx, bins=50, color='blue', alpha=0.7, label='Batch Means')
plt.axvline(mean_of_means_Wx, color='red', linestyle='dashed', linewidth=2, label='Mean')
for z in z_scores:
    plt.axvline(mean_of_means_Wx + z * std_Wx, color='green', linestyle='dashed', linewidth=2, label=f'+{z} Std Dev')
    plt.axvline(mean_of_means_Wx - z * std_Wx, color='green', linestyle='dashed', linewidth=2, label=f'-{z} Std Dev')

plt.title('Monte Carlo Simulation of Batch Means')
plt.xlabel('Mean of Batches')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()




