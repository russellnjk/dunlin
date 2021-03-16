import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import matplotlib.ticker as mtick

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin            as dn
import dunlin.simulation as sim

plt.style.use(dn.styles['dark_style_multi'])
plt.close('all')

filename = 'TestRes_1.ini'

model_data, sim_args = sim.read_ini(filename)

coarse_1   = model_data['coarse_1']['model']

coarse_1s  = list(coarse_1.states) 
coarse_1o  = list(model_data['coarse_1']['objectives'].keys())
coarse_1a  = ['x', 'S', 'P', 'R'] + coarse_1o


plot_index = {'coarse_1': coarse_1a}

simulation_results   = sim.integrate_models(sim_args)
figs, AX             = sim.plot_simulation_results(plot_index, 
                                                   simulation_results,
                                                   label       = None
                                                   )
#Post procesing
for fig in figs:
    for ax in fig.axes:
        ax.legend()
        ax.ticklabel_format(style='sci', scilimits=(-2, 3))
# AX['coarse_1']['Vcell'].set_ylim(0, 5e-15)
AX['coarse_1']['r_sat'].set_ylim(0, 1)
AX['coarse_1']['reg'].set_ylim(0, 1)
