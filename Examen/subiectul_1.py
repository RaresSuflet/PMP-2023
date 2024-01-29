import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

def functie_subiect1():
    #a incarc datele din csv si fac variabile pt coloanele care ma intereseaza
    Housing = pd.read_csv("BostonHousing.csv")
    medv = Housing['medv'].values
    rm = Housing['rm'].values
    crim = Housing['crim'].values
    indus = Housing['indus'].values
    #b modelul unde am ales pentru fiecare variabila independenta o distribuitie normala, iar medv in functie de cele
    # independente
    with pm.Model() as Model:
        alfa = pm.Normal('alfa', mu=0, sigma=10)
        beta_rm = pm.Normal('beta_rm', mu=0, sigma=10)
        beta_crim = pm.Normal('beta_crim', mu=0, sigma=10)
        beta_indus = pm.Normal('beta_indus', mu=0, sigma=10)
        mu = alfa + beta_rm * rm + beta_crim * crim + beta_indus + indus

        medv_pred = pm.Normal('medv_pred', mu=mu, sigma=10, observed=medv)
        idata = pm.sample(2000, tune=1000)
    #c estimari 95% ale parametrilor, parametrul care influenteaza cel mai mult este proprotia suprafetei comerciale
    az.plot_forest(idata, hdi_prob=0.95)
    plt.show()
    az.summary(idata, hdi_prob=0.95, var_names=["beta_rm", "beta_crim", "beta_indus"])

    #d extrageri din distributia predictiva posterioara si gasim cu ele intervalul de predictie 50%
    post_pred = pm.sample_posterior_predictive(idata, model=Model)
    y_post_pred = post_pred.posterior_predictive['medv_pred'].stack(sample=("chain", "draw")).values
    az.plot_posterior(y_post_pred, hdi_prob=0.5)
    plt.show()


if __name__ == "__main__":
    functie_subiect1()


