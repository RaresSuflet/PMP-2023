import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

procesare_servere = [(4,3), (4,2), (5,2), (5,3)]
prob_servere = [0.25, 0.25, 0.30, 0.20]

latenta = 4

timp_servire_servere = []

for procesare in procesare_servere:
    forma, scala = procesare
    timp_servire_server = stats.gamma.rvs(forma, scale=scala, size=int(10000 * prob_servere.pop(0)))
    timp_latenta = stats.expon.rvs(scale=1/latenta, size=len(timp_servire_server))
    timp_total = timp_servire_server + timp_latenta
    timp_servire_servere.extend(timp_total)

timp_servire_servere = np.array(timp_servire_servere)
pb = np.mean(timp_servire_servere > 3)
print("Prob ca timpul sa fie mai mare de 3 milisecunde", pb)
az.plot_dist(timp_servire_servere)
plt.show()
