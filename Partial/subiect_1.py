from scipy import stats
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
counter_p0_castig = 0
counter_p1_castig = 0
#1
for j in range(0, 20000):
    moneda_alegere_p0 = stats.binom.rvs(1, 0.5, size=1) # determinam ce jucator va fi primul
    if moneda_alegere_p0[0] == 1: # a fost ales p0 ca primul jucator
        stema_moneda_p0 = stats.binom.rvs(1, 0.3, size=1) # p0 cu moneda masluita
        n = stema_moneda_p0[0]
        stema_moneda_p1 = stats.binom.rvs(1, 0.5, size=n + 1)
        m = 0
        if n != 0: # m contorul pentru de cate ori s-a obtinut la a doua runda stema
            for i in range(0, n + 1):
                if stema_moneda_p1[i] == 1:
                    m += 1
        else:
            if stema_moneda_p1[0] == 1:
                m += 1
    else: #a fost ales p1 ca primul jucator
        stema_moneda_p1 = stats.binom.rvs(1, 0.5, size=1)
        n = stema_moneda_p1[0]
        stema_moneda_p0 = stats.binom.rvs(1, 0.3, size=n + 1)
        m = 0
        if n != 0:
            for i in range(0, n + 1):
                if stema_moneda_p0[i] == 1:
                    m += 1
        else:
            if stema_moneda_p0[0] == 1:
                m += 1
    if n >= m:
        if moneda_alegere_p0[0] == 1:
            counter_p0_castig += 1
        else:
            counter_p1_castig += 1
    else:
        if moneda_alegere_p0[0] == 1:
            counter_p1_castig += 1
        else:
            counter_p0_castig += 1
if counter_p0_castig > counter_p1_castig: #
    print(f"p0 are sanse mai mari de castig, {counter_p0_castig}, {counter_p1_castig}")
else:
    print(f"p1 are sanse mai mari de castig, {counter_p1_castig}, {counter_p0_castig}")

#2
coin_game = BayesianNetwork([('moneda_alegere', 'stema_prima_runda'), ('stema_prima_runda,', 'stema_a_doua_runda')])
moneda_alegere = TabularCPD(variable='moneda_alegere', variable_card=2, values=[[0.5], [0.5]])
stema_prima_runda = TabularCPD(variable='stema_prima_runda', variable_card=2, values=[[0.5, 0.7], [0.5, 0.3]],
                               evidence=['moneda_alegere'], evidence_card=[2])
# stema_a_doua_runda = TabularCPD(variable='stema_a_doua_runda', variable_card=)
