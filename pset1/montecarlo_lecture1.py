import numpy as np
from scipy.stats import norm


# Function to perform Monte Carlo simulation
def monte_carlo_simulation(true_mean, true_std, sample_size, alpha, num_simulations):
    coverage_known_std = 0
    coverage_estimated_std = 0

    for _ in range(num_simulations):
        # Step 1: Draw samples from a normal distribution
        samples = np.random.normal(true_mean, true_std, sample_size)

        # Step 2: Construct confidence intervals
        # With known standard deviation
        z_critical_known_std = norm.ppf(1 - (alpha / 2))
        margin_known_std = z_critical_known_std * (true_std / np.sqrt(sample_size))
        ci_known_std = (
            np.mean(samples) - margin_known_std,
            np.mean(samples) + margin_known_std,
        )

        # With estimated standard deviation
        # Use ddof=1 for unbiased estimate
        sample_std = np.std(samples, ddof=1)
        z_critical_estimated_std = norm.ppf(1 - (alpha / 2))
        margin_estimated_std = z_critical_estimated_std * (
            sample_std / np.sqrt(sample_size)
        )
        ci_estimated_std = (
            np.mean(samples) - margin_estimated_std,
            np.mean(samples) + margin_estimated_std,
        )

        # Step 3: Check if each interval covers the true mean
        if ci_known_std[0] <= true_mean <= ci_known_std[1]:
            coverage_known_std += 1
        if ci_estimated_std[0] <= true_mean <= ci_estimated_std[1]:
            coverage_estimated_std += 1

    # Calculate coverage percentages
    coverage_known_std_percent = coverage_known_std / num_simulations * 100
    coverage_estimated_std_percent = coverage_estimated_std / num_simulations * 100

    return coverage_known_std_percent, coverage_estimated_std_percent


# Parameters
true_mean = 0
true_std = 1
num_simulations = 1000
alpha = 0.05
sample_sizes = [30, 100, 500]

# Perform simulations for different sample sizes
for sample_size in sample_sizes:
    coverage_known_std_percent, coverage_estimated_std_percent = monte_carlo_simulation(
        true_mean, true_std, sample_size, alpha, num_simulations
    )
    print(f"Sample Size: {sample_size}")
    print(f"Coverage with Known Std: {coverage_known_std_percent}%")
    print(f"Coverage with Estimated Std: {coverage_estimated_std_percent}%")
    print("\n")
