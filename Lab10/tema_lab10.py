import pymc as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az


# ex1
def main_ex1():
    az.style.use('arviz-darkgrid')
    dummy_data = np.loadtxt('dummy.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    order = 5
    x_1p = np.vstack([x_1**i for i in range(1, order+1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')

    with pm.Model() as model_p:
        alfa = pm.Normal('alfa', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=order)
        epsilon = pm.HalfNormal('epsilon', 5)
        mu = alfa + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)

    # b
    with pm.Model() as model_p_sd_100:
        alfa = pm.Normal('alfa', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=100, shape=order)
        epsilon = pm.HalfNormal('epsilon', 5)
        mu = alfa + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_1s)
        idata_p_sd_100 = pm.sample(2000, return_inferencedata=True)

    array_sd = np.array([10, 0.1, 0.1, 0.1, 0.1])
    with pm.Model() as model_p_sd_array:
        alfa = pm.Normal('alfa', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=array_sd, shape=order)
        epsilon = pm.HalfNormal('epsilon', 5)
        mu = alfa + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_1s)
        idata_p_sd_array = pm.sample(2000, return_inferencedata=True)

    x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
    alfa_p_post = idata_p.posterior['alfa'].mean(("chain", "draw")).values
    beta_p_post = idata_p.posterior['beta'].mean(("chain", "draw")).values
    idx_p = np.argsort(x_1s[0])
    y_p_post = alfa_p_post + np.dot(beta_p_post, x_1s)
    plt.plot(x_1s[0][idx_p], y_p_post[idx_p], 'C1', label=f'model polinomial order=5')

    alfa_sd_100_post = idata_p_sd_100.posterior['alfa'].mean(("chain", "draw")).values
    beta_sd_100_post = idata_p_sd_100.posterior['beta'].mean(("chain", "draw")).values
    idx_sd_100 = np.argsort(x_1s[0])
    y_sd_100_post = alfa_sd_100_post + np.dot(beta_sd_100_post, x_1s)
    plt.plot(x_1s[0][idx_sd_100], y_sd_100_post[idx_sd_100], 'C2', label=f'model sd=100')

    alfa_sd_array_post = idata_p_sd_array.posterior['alfa'].mean(("chain", "draw")).values
    beta_sd_array_post = idata_p_sd_array.posterior['beta'].mean(("chain", "draw")).values
    idx_sd_array = np.argsort(x_1s[0])
    y_sd_array_post = alfa_sd_array_post + np.dot(beta_sd_array_post, x_1s)
    plt.plot(x_1s[0][idx_sd_array], y_sd_array_post[idx_sd_array], 'C3', label=f'model sd=array')

    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()
    return x_1


# 2
def main_ex2(x_1):
    order = 5
    x_500 = np.random.uniform(low=min(x_1), high=max(x_1), size=500)
    y_500 = x_500 ** 2 - 0.64 + np.random.normal(0, 1, size=500)
    x_500p = np.vstack([x_500**i for i in range(1, order+1)])
    x_500s = (x_500p - x_500p.mean(axis=1, keepdims=True)) / x_500p.std(axis=1, keepdims=True)
    y_500s = (y_500 - y_500.mean()) / y_500.std()
    plt.scatter(x_500s[0], y_500s)
    plt.xlabel('x_500')
    plt.ylabel('y_500')

    with pm.Model() as model_p:
        alfa = pm.Normal('alfa', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=order)
        epsilon = pm.HalfNormal('epsilon', 5)
        mu = alfa + pm.math.dot(beta, x_500s)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_500s)
        idata_p = pm.sample(2000, return_inferencedata=True)

    with pm.Model() as model_p_sd_100:
        alfa = pm.Normal('alfa', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=100, shape=order)
        epsilon = pm.HalfNormal('epsilon', 5)
        mu = alfa + pm.math.dot(beta, x_500s)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_500s)
        idata_p_sd_100 = pm.sample(2000, return_inferencedata=True)

    array_sd = np.array([10, 0.1, 0.1, 0.1, 0.1])
    with pm.Model() as model_p_sd_array:
        alfa = pm.Normal('alfa', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=array_sd, shape=order)
        epsilon = pm.HalfNormal('epsilon', 5)
        mu = alfa + pm.math.dot(beta, x_500s)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_500s)
        idata_p_sd_array = pm.sample(2000, return_inferencedata=True)

    x_new = np.linspace(x_500s[0].min(), x_500s[0].max(), 100)
    alfa_p_post = idata_p.posterior['alfa'].mean(("chain", "draw")).values
    beta_p_post = idata_p.posterior['beta'].mean(("chain", "draw")).values
    idx_p = np.argsort(x_500s[0])
    y_p_post = alfa_p_post + np.dot(beta_p_post, x_500s)
    plt.plot(x_500s[0][idx_p], y_p_post[idx_p], 'C1', label=f'model polinomial order=5')

    alfa_sd_100_post = idata_p_sd_100.posterior['alfa'].mean(("chain", "draw")).values
    beta_sd_100_post = idata_p_sd_100.posterior['beta'].mean(("chain", "draw")).values
    idx_sd_100 = np.argsort(x_500s[0])
    y_sd_100_post = alfa_sd_100_post + np.dot(beta_sd_100_post, x_500s)
    plt.plot(x_500s[0][idx_sd_100], y_sd_100_post[idx_sd_100], 'C2', label=f'model sd=100')

    alfa_sd_array_post = idata_p_sd_array.posterior['alfa'].mean(("chain", "draw")).values
    beta_sd_array_post = idata_p_sd_array.posterior['beta'].mean(("chain", "draw")).values
    idx_sd_array = np.argsort(x_500s[0])
    y_sd_array_post = alfa_sd_array_post + np.dot(beta_sd_array_post, x_500s)
    plt.plot(x_500s[0][idx_sd_array], y_sd_array_post[idx_sd_array], 'C3', label=f'model sd=array')

    plt.scatter(x_500s[0], y_500s, c='C0', marker='.')
    plt.legend()
    plt.show()


# 3
def main_ex3():
    dummy_data = np.loadtxt('dummy.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    order = 3
    x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')

    with pm.Model() as model_p:
        alfa = pm.Normal('alfa', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=order)
        epsilon = pm.HalfNormal('epsilon', 5)
        mu = alfa + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)

    x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
    alfa_p_post = idata_p.posterior['alfa'].mean(("chain", "draw")).values
    beta_p_post = idata_p.posterior['beta'].mean(("chain", "draw")).values
    idx_p = np.argsort(x_1s[0])
    y_p_post = alfa_p_post + np.dot(beta_p_post, x_1s)
    plt.plot(x_1s[0][idx_p], y_p_post[idx_p], 'C1', label=f'model cubic order=3')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    x_1 = main_ex1()
    main_ex2(x_1)
    main_ex3()
