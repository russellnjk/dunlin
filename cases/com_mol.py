import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin               as dn
import dunlin.combinatorial as com

plt.style.use(dn.styles['dark_style_multi'])
plt.close('all')

filename = 'com_mol.ini'

model_data, sim_args            = com.read_ini(filename)
simulation_results, obj_results = com.evaluate_objectives(sim_args, 'max')

model_1       = sim_args['model_1']['model']
obj_1, best_1 = obj_results['model_1']
print(obj_1)
print(best_1)