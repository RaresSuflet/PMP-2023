import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
def main():
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

        timp_total = pm.math.sum(timp_plasare_plata[:numar_clienti]) + pm.math.sum(timp_pregatire[:numar_clienti])
    with model:
        trace = pm.sample(1000)

    # df = trace.to_dataframe(trace)
    az.plot_posterior(trace)
    plt.show()

# pentru datele de la ex 2
# P(X<15) = 1 - exp(-15/alfa)
# 1 - exp(-15/alfa) = 0.95
# 0.05 = exp(-15/alfa)
# ln(0.05) = -15/alfa
# alfa = - 15 / (ln(0.05))
def alpha_max():
    alpha_maxim = -15 / np.log(0.05)
    return alpha_maxim
if __name__ == '__main__':
    main()
    print(alpha_max())
