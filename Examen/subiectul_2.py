import numpy as np
import scipy.stats as stats

def posterior_grid(grid_points=50, heads=6, tails=9):
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1 / grid_points, grid_points)
    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

def functie_subiect_2():

if __name__ == "__main__":
    posterior_grid()
