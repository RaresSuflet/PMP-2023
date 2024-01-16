import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

centered_eight = az.load_arviz_data("centered_eight")
non_centered_eight = az.load_arviz_data("non_centered_eight")

# 1
print("MODEL CENTRAT")
print("Numarul de lanturi:", centered_eight.posterior.chain.size)
print("Marimea totala a esantionului generat:", centered_eight.posterior.draw.size)
az.plot_posterior(centered_eight, var_names=["mu", "tau"])
plt.show()

print("\nMODEL NECENTRAT")
print("Numarul de lanturi:", non_centered_eight.posterior.chain.size)
print("Marimea totala a esantionului generat:", non_centered_eight.posterior.draw.size)
az.plot_posterior(non_centered_eight, var_names=["mu", "tau"])
plt.show()

# 2
# Rhat
centered_eight_summary = az.summary(centered_eight, var_names=["mu", "tau"])
non_centered_eight_summary = az.summary(non_centered_eight, var_names=["mu", "tau"])
summaries = pd.concat([centered_eight_summary, non_centered_eight_summary])
summaries.index = ['centered_mu', 'centered_tau', 'non_centered_mu', 'non_centered_tau']
print(summaries.to_string())
# autocorelatie
az.plot_autocorr(centered_eight, var_names=["mu", "tau"])
plt.show()

az.plot_autocorr(non_centered_eight, var_names=["mu", "tau"])
plt.show()

#3
divergences_centered = centered_eight.sample_stats.diverging.sum()
divergences_non_centered = non_centered_eight.sample_stats.diverging.sum()

print("\nNumarul de divergente in modelul centrat:", divergences_centered.values.item())
print("\nNumarul de divergente in modelul necentrat", divergences_non_centered.values.item())

az.plot_pair(centered_eight, var_names=["mu", "tau"], divergences=True)
plt.show()
az.plot_parallel(centered_eight)
plt.show()

az.plot_pair(non_centered_eight, var_names=["mu", "tau"], divergences=True)
plt.show()
az.plot_parallel(non_centered_eight)
plt.show()
