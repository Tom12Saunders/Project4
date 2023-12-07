import numpy as np
import matplotlib.pyplot as plt


# Analytical solution for comparison
def analytical_solution(Nc, xB, ST, SS, Q, delta_x):  # Added delta_x as a parameter
    Sigma_a = ST - SS
    alpha = np.sqrt(Sigma_a * ST)
    a = -ST * Q / (Sigma_a * (ST * np.cosh(alpha * xB) + alpha * np.sinh(alpha * xB)))
    x_values = np.linspace(0, xB, Nc) + delta_x / 2  # Center of each cell for comparison
    return a * np.cosh(alpha * x_values) + Q / Sigma_a


def monte_carlo_simulation(Nc, Nb, Np, ST, SS, Q, xB, delta_x):
    fluxes = np.zeros(Nc)
    leakages = np.zeros(2)  # [left_boundary_leakage, right_boundary_leakage]
    fluxes_squared = np.zeros(Nc)
    leakages_squared = np.zeros(2)

    for _ in range(Nb):
        batch_fluxes = np.zeros(Nc)
        batch_leakages = np.zeros(2)

        for _ in range(Np):
            x = 0  # Start at x = 0
            mu = np.random.choice([-1, 1])  # Randomly choose initial direction

            while True:
                distance = -np.log(np.random.rand()) / ST
                x_new = x + mu * distance

                # Reflective boundary condition at x=0
                if x_new < 0:
                    mu = -mu  # Reverse direction
                    x_new = -x_new

                # Check if the particle has left the system
                if x_new >= xB:
                    if mu > 0:  # Particle moving to the right
                        batch_leakages[1] += Q / Np  # Full leakage for particles exiting the right boundary
                    break  # Particle leaves the system, stop tracking it
                else:
                    # Ensure that the particle does not skip cells
                    while x < x_new:
                        cell_index = int(x / delta_x)
                        next_cell_boundary = (cell_index + 1) * delta_x
                        step_to_boundary = min(next_cell_boundary - x, x_new - x)

                        # Calculate contribution to the scalar flux in the current cell
                        batch_fluxes[cell_index] += step_to_boundary * Q / (delta_x * Np)
                        x += step_to_boundary  # Move the particle to the new position
                        if x >= x_new:
                            break  # The particle has reached its new position

                        # Scatter particle at cell boundary
                        mu = np.random.choice([-1, 1])  # Randomly choose new direction after scattering

        # Accumulate batch results
        fluxes += batch_fluxes
        fluxes_squared += batch_fluxes ** 2
        leakages += batch_leakages
        leakages_squared += batch_leakages ** 2

    # Compute averages and standard deviations
    avg_fluxes = fluxes / Nb
    avg_leakages = leakages / Nb
    avg_fluxes_squared = fluxes_squared / Nb
    avg_leakages_squared = leakages_squared / Nb

    # Corrected standard deviation calculation
    std_dev_fluxes = np.sqrt((avg_fluxes_squared - avg_fluxes ** 2) / (Nb - 1))
    std_dev_leakages = np.sqrt((avg_leakages_squared - avg_leakages ** 2) / (Nb - 1))

    # Compute relative deviations
    percent_rel_dev_fluxes = 100 * std_dev_fluxes / avg_fluxes
    percent_rel_dev_leakages = 100 * std_dev_leakages / avg_leakages

    # Handle potential divide by zero
    percent_rel_dev_fluxes = np.nan_to_num(percent_rel_dev_fluxes, nan=0.0)
    percent_rel_dev_leakages = np.nan_to_num(percent_rel_dev_leakages, nan=0.0)

    return avg_fluxes, avg_leakages, percent_rel_dev_fluxes, percent_rel_dev_leakages



# Parameters
Nc = 10
Nb = 10000
Np = 1
ST = 1.0
SS = 0.5
Q = 1.0
xB = 3.0
delta_x = xB / Nc

# Running the Monte Carlo simulation
avg_fluxes, avg_leakages, percent_rel_dev_fluxes, percent_rel_dev_leakages = monte_carlo_simulation(
    Nc, Nb, Np, ST, SS, Q, xB, delta_x)

# Analytical solution - now passing delta_x
analytic_fluxes = analytical_solution(Nc, xB, ST, SS, Q, delta_x)

# Plotting
x_values = np.linspace(0, xB, Nc)
plt.plot(x_values, avg_fluxes, label='Monte Carlo Flux')
plt.plot(x_values, analytic_fluxes, label='Analytic Flux', linestyle='--')
plt.title('Comparison of Monte Carlo and Analytic Scalar Flux')
plt.xlabel('Spatial Domain (x)')
plt.ylabel('Scalar Flux')
plt.legend()
plt.grid(True)
plt.show()

# Printing results
print("Final Fluxes:", avg_fluxes)
print("Percent Relative Deviation in Fluxes:", percent_rel_dev_fluxes)
print("Final Leakages:", avg_leakages)
print("Percent Relative Deviation in Leakages:", percent_rel_dev_leakages)

# Print the final fluxes and their percent relative deviations
print("Cell-by-Cell Comparison of Final Fluxes:")
for i, (avg_flux, percent_dev_flux, analytic_flux) in enumerate(zip(avg_fluxes, percent_rel_dev_fluxes, analytic_fluxes)):
    print(f"Cell {i+1}: Monte Carlo Flux = {avg_flux:.4f}, Analytic Flux = {analytic_flux:.4f}, "
          f"Percent Relative Deviation = {percent_dev_flux:.2f}%")

# Print the final leakages and their percent relative deviations
print("\nLeakages Comparison:")
leakage_positions = ['Left Boundary', 'Right Boundary']
for pos, (leakage, percent_dev_leakage) in zip(leakage_positions, zip(avg_leakages, percent_rel_dev_leakages)):
    print(f"{pos} Leakage: Monte Carlo Leakage = {leakage:.4f}, "
          f"Percent Relative Deviation = {percent_dev_leakage:.2f}%")

# Comparison plot for final fluxes
plt.plot(x_values, avg_fluxes, label='Monte Carlo Flux')
plt.plot(x_values, analytic_fluxes, label='Analytic Flux', linestyle='--')
plt.title('Comparison of Monte Carlo and Analytic Scalar Flux')
plt.xlabel('Spatial Domain (x)')
plt.ylabel('Scalar Flux')
plt.legend()
plt.grid(True)
plt.show()