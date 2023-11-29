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
        sigma = pm.HalfNormal('sigma', sigma=1)
        mu = alpha + beta1 * x1 + beta2 * x2

        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
        trace = pm.sample(5000, tune=2500, cores=1)
    az.plot_posterior(trace, hdi_prob=0.95)
    plt.show()
    # 2
    hdi_beta1 = az.hdi(trace.posterior['beta1'].values.flatten(), hdi_prob=0.95)
    hdi_beta2 = az.hdi(trace.posterior['beta2'].values.flatten(), hdi_prob=0.95)
    print(f"95% HDI pt beta1: {hdi_beta1}")
    print(f"95% HDI pt beta2: {hdi_beta2}")
    # 3
    if 0 not in hdi_beta1 and 0 not in hdi_beta2:
        print("Frecventa procesorului si marimea hard diskului sunt predictori utili pentru pretul de vanzare")
    else:
        print("Frecventa procesorului si marimea hard diskului nu sunt predictori utili pentru pretul de vanzare")

    processor_freq = 33
    hard_disk_size = np.log(540)
    # 4
    mu_sim = trace.posterior['alpha'].values + trace.posterior['beta1'].values * processor_freq + trace.posterior[
        'beta2'].values * hard_disk_size
    hdi_mu_sim = az.hdi(mu_sim.flatten(), hdi_prob=0.9)
    print(f"Interval HDI pentru pretul de vanzare asteptat: {hdi_mu_sim}")
    

if __name__ == "__main__":
    main()


