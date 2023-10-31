import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
data = pd.read_csv('trafic.csv')

minute = data['minut'].tolist()
numar_masini = data['nr. masini'].values

ore_crestere = [7, 16]
ore_descrestere = [8, 19]

total_intervale = len(minute)
model = pm.Model()
with model:
    lambda_poisson = pm.Normal('lambda_poisson', mu=10, sigma=5)
    trafic = pm.Poisson('trafic', mu=lambda_poisson, observed=numar_masini)

    for hour in ore_crestere:
        lambda_poisson = pm.Deterministic(f'lambda_poisson{hour}', lambda_poisson * 1.2)
    for hour in ore_descrestere:
        lambda_poisson = pm.Deterministic(f'lambda_poisson{hour}', lambda_poisson * 0.8)

with model:
    trace = pm.sample(2000)


df = trace.to_dataframe(trace)


az.plot_posterior(trace)
plt.show()


