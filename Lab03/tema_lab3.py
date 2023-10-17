from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

import matplotlib.pyplot as plt
import networkx as nx

model = BayesianNetwork([('C', 'I'), ('C', 'A'), ('I', 'A')])

cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.9995], [0.0005]]) # din procente in probabilitati
                                                                               # C=0 sa nu aiba loc cutremur
                                                                               # C=1 sa aiba loc cutremur
cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.99, 0.97], [0.01, 0.03]], evidence=['C'], evidence_card=[2])
                                                                               # I = 0 sa nu aiba loc incendiu
                                                                               # I=1 sa aiba loc incendiu
cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.9999, 0.05, 0.98, 0.02], [0.0001, 0.95, 0.02, 0.98]],
                   evidence=['I', 'C'], evidence_card=[2, 2])
                                                                                # A = 0 sa nu se declanseze alarma
                                                                                # A=1 sa se declanseze alarma
model.add_cpds(cpd_c, cpd_i, cpd_a)

assert model.check_model()

infer = VariableElimination(model)

result = infer.query(variables=['A'], evidence={'C': 1, 'I': 1})
print(result)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()

#ex 2 probabilitatea sa fi avut loc un cutremur, stiind ca alarma a fost declansata P(C=1|A=1)


#ex 3 probabilitatea sa fi avut loc un incendiu, fara ca alarma sa se declanseze P(I=1|A=0)