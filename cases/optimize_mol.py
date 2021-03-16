import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin            as dn
import dunlin.simulation as sim
import dunlin.tuning     as tun

plt.style.use(dn.styles['dark_style_multi'])
plt.close('all')

filename = 'opt_mol.ini'

#Combinatorial simulation and objective evaluation
model_data, sim_args = sim.read_ini(filename)
obj_results          = tun.evaluate_objectives(sim_args)

print(obj_results['model_1'])
