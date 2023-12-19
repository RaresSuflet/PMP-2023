import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


def main():
    #1
    clusters = 3
    n_cluster = [200, 150, 100]
    n_total = sum(n_cluster)
    means = [5, 0, 2.5]
    std_devs = [2, 2, 1]
    mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))
    az.plot_kde(np.array(mix))
    #2
    clusters = [2, 3, 4]

    models = []
    idatas = []
    for cluster in clusters:
        with pm.Model() as model:
            p = pm.Dirichlet('p', a=np.ones(cluster))
            means = pm.Normal('means',
                              mu=np.linspace(mix.min(), mix.max(), cluster),
                              sigma=10, shape=cluster,
                              transform=pm.distributions.transforms.ordered)
            sd = pm.HalfNormal('sd', sigma=10)
            y = pm.NormalMixture('y', w=p, mu=means, sigma=sd, observed=mix)
            idata = pm.sample(2000, tunes=1000, target_accept=0.95, random_seed=123, return_inferencedata=True)
            idatas.append(idata)
            models.append(model)
    #3
    for i in range(0, 3):
        pm.compute_log_likelihood(idatas[i], model=models[i])
    traces = [idata.posterior for idata in idatas]
    comp_waic = az.compare({'model_2': traces[0], 'model_3': traces[1], 'model_4': traces[2]},
                           method='BB-pseudo-BMA', ic="waic", scale="deviance")
    az.plot_compare(comp_waic)
    plt.show()
    comp_loo = az.compare({'model_2': traces[0], 'model_3': traces[1], 'model_4': traces[2]},
                          method='BB-pseudo-BMA', ic="loo", scale="deviance")
    az.plot_compare(comp_loo)
    plt.show()


if __name__ == '__main__':
    main()
