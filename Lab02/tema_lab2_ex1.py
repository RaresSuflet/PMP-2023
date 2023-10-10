import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

#incomplet
np.random.seed(1)

lambda1 = 4
lambda2 = 6
prob_mecanic1 = 0.4

primul_mecanic = stats.expon.rvs(scale=1/lambda1, size=10000)
aldoilea_mecanic = stats.expon.rvs(scale=1/lambda2, size=10000)



az.plot_posterior({'primul_mecanic':primul_mecanic, 'aldoilea_mecanic':aldoilea_mecanic})
plt.show()