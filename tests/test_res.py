import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin            as dn
import dunlin.simulation as sim

plt.style.use(dn.styles['dark_style_multi'])
plt.close('all')

filename = 'res_1.ini'

model_data, sim_args = sim.read_ini(filename)

weis   = model_data['weisse']['model'] 
weis_s = list(weis.states)

plot_index = {'weisse': weis_s}
color      = {'weisse': {0: sim.colors['cobalt']
                          }
              }

AX = None

# weis.solver_args['rtol'] = 1e-9
# weis.solver_args['full_output'] = True

simulation_results   = sim.integrate_models(sim_args)
figs, AX             = sim.plot_simulation_results(plot_index, 
                                                    simulation_results,
                                                    AX          = AX,
                                                    color       = color,
                                                    label       = 'scenario'
                                                    )