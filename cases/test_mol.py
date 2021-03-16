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

filename = 'TestMol_1.ini'

model_data, sim_args = sim.read_ini(filename)

model_1   = model_data['model_1']['model']
model_1s  = list(model_1.states)
model_1x  = model_1s[:5]
model_1y  = model_1s[5:]
model_1ox = ['cx_0', 'cx_10', 'cx_20', 'cx_40', 'cx_final']
model_1oy = ['cy_0', 'cy_10', 'cy_20', 'cy_40', 'cy_final']
model_1v  = model_1s + model_1ox + model_1oy

#Figs and Axes
fig = plt.figure()
# AX_ = [fig.add_subplot(5, 1, i+1) for i in range(5)]
AX_ = [fig.add_subplot(2, 2, 1)]*len(model_1x ) + \
      [fig.add_subplot(2, 2, 2)]*len(model_1y ) + \
      [fig.add_subplot(2, 2, 3)]*len(model_1ox) + \
      [fig.add_subplot(2, 2, 4)]*len(model_1oy)
AX  = {'model_1': dict(zip(model_1v, AX_))}

#Plot settings
plot_index = {'model_1': model_1v }
s_colors_1 = sim.palette_types['light'](len(model_1ox), color=sim.colors['crimson'])
s_colors_2 = sim.palette_types['light'](len(model_1ox), color=sim.colors['purple' ])
o_colors_1 = sim.palette_types['light'](len(model_1ox), color=sim.colors['cobalt' ])
o_colors_2 = sim.palette_types['light'](len(model_1oy), color=sim.colors['teal'   ])
v_colors_1 = dict(zip(model_1v, s_colors_1 + s_colors_1 + o_colors_1 + o_colors_1))
v_colors_2 = dict(zip(model_1v, s_colors_2 + s_colors_2 + o_colors_2 + o_colors_2))
color      = {'model_1': {0: {0 : v_colors_1
                              },
                          1: {0: v_colors_2
                              }
                          }
                }

simulation_results   = sim.integrate_models(sim_args)
figs, AX             = sim.plot_simulation_results(plot_index, 
                                                   simulation_results,
                                                   AX          = AX,
                                                   color       = color,
                                                   label       = 'scenario'
                                                   )

#Post processing
sim.fs(fig)
for ax in AX_:
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')