import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

def read_data():
    data = pd.read_csv('Prices.csv')
    x1 = data['Speed']
    x2 = np.log(data['HardDrive'])
    y = data['Price']
    return np.array(x1), np.array(x2), np.array(y)


def main():
    x1, x2, y = read_data()
    with pm.Model() as regression_model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=10)
        beta2 = pm.Normal('beta2', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=10)
        mu = alpha + beta1 * x1 + beta2 * x2

        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
        trace = pm.sample(5000, tune=1000, cores=1)
    az.plot_posterior(trace)
    plt.show()


if __name__ == "__main__":
    main()


