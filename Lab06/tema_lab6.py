import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

fig, axis = plt.subplots(len(Y_values), len(theta_values), figsize=(10, 10))

for i, Y in enumerate(Y_values):
    for j, theta in enumerate(theta_values):
        with pm.Model() as model:
            n = pm.Poisson('n', mu=10)

            y = pm.Binomial('y', n=n, p=theta, observed=Y)

            trace = pm.sample(20000, cores=1)

        az.plot_posterior(trace, ax=axis[i, j])
        axis[i, j].set_title(f'Y = {Y}, θ = {theta}')

plt.tight_layout()
plt.show()
