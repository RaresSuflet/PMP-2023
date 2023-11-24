import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
with pm.Model() as model:
    mu = pm.Uniform('mu', lower=0, upper=300) # secunde estimam ca limita ar fi undeva la 5 minute maxim
    sigma = pm.HalfNormal('sigma', sigma=20) #sigma 20 ar trebui sa fie o valoare suficient de nare
    timp_mediu = pm.Normal("timp_mediu", mu=mu, sigma=sigma) # 
    trace = pm.sample(200, cores=1)
    az.plot_posterior(trace)
    plt.show()
