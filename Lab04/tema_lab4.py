import pymc as pm
import numpy as np

model = pm.Model()
with model:
    lambda_poisson = 20
    numar_clienti = pm.Poisson('numar clienti', lambda_poisson)

    media_plasare_plata = 2
    dev_standard_plasare_plata = 0.5
    timp_plasare_plata = pm.Normal('timp plasare plata', mu=media_plasare_plata, tau=1.0 / (dev_standard_plasare_plata ** 2),
                                    size=50)

    alpha = 5  # ales arbitrar
    timp_pregatire = pm.Exponential('timp_pregatire', lam=1.0/alpha, size=50)

    timp_total = timp_plasare_plata + timp_pregatire
with model:
    trace = pm.sample(1000)
