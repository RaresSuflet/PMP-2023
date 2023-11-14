import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
# a
df = pd.read_csv("auto-mpg.csv", header=0)
df = df.dropna()
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df = df.dropna(subset=['horsepower'])
plt.scatter(df['horsepower'], df['mpg'])
plt.title('Relatia dintre CP si mpg')
plt.xlabel('Cai putere (CP)')
plt.ylabel('Mile/galon (mpg)')
plt.show()

x = df['horsepower'].values
y = df['mpg'].values
# b
with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=1)
    mu = pm.Deterministic('mu', alpha + beta * x)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=1, observed=y)
with model:
    trace = pm.sample(1000, tune=1000, cores=1, return_inferencedata=True)
az.plot_trace(trace)
plt.show()

# c
plt.plot(x, y, 'C0.')
posterior_g = trace.posterior.stack(samples={"chain", "draw"})
alpha_mean = posterior_g['alpha'].mean().item()
beta_mean = posterior_g['beta'].mean().item()
draws = range(0, posterior_g.samples.size, 10)
plt.plot(x, posterior_g['alpha'][draws].values + posterior_g['beta'][draws].values * x[:,None], c='gray', alpha=0.5)
plt.plot(x, alpha_mean + beta_mean * x, c='k', label=f'y = {alpha_mean:.2f} + {beta_mean:.2f} * x')
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend()
plt.show()