import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
data = pd.read_csv('Admission.csv')

gre = np.array(data['GRE'].values)
gpa = np.array(data['GPA'].values)
admission = np.array(data['Admission'].values)

with pm.Model() as logistic_model:
    beta0 = pm.Normal('beta0', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)

    p = pm.math.sigmoid(beta0 + beta1 * gre + beta2 * gpa)

    admission_observed = pm.Bernoulli('admission_observed', p ,observed=admission)

with logistic_model:
    trace = pm.sample(2000, tune=1000, cores=1)

az.plot_trace(trace)
plt.show()
