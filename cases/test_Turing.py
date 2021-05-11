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

#Use any file with TestTuring
filename = 'TestTuring_1.ini'

model_data, sim_args = sim.read_ini(filename)

model_1   = model_data['model_1']['model']
model_1s  = list(model_1.states)
model_1x  = model_1s[:len(model_1s)//2]
model_1y  = model_1s[len(model_1s)//2:]
model_1ex = ['cx_0', 'cx_20', 'cx_40', 'cx_final']
model_1ey = ['cy_0', 'cy_20', 'cy_40', 'cy_final']
model_1v  = model_1s + model_1ex + model_1ey

#Figs and Axes
fig = plt.figure()
AX_ = [fig.add_subplot(2, 2, 1)]*len(model_1x ) + \
      [fig.add_subplot(2, 2, 2)]*len(model_1y ) + \
      [fig.add_subplot(2, 2, 3)]*len(model_1ex) + \
      [fig.add_subplot(2, 2, 4)]*len(model_1ey)
AX  = {'model_1': dict(zip(model_1v, AX_))}

#Plot settings
plot_index = {'model_1': model_1v }
palette    = ['purple', 'cobalt', 'teal']
s_colors_1 = sim.palette_types['light'](len(model_1x), color=sim.colors['purple'])
s_colors_2 = sim.palette_types['light'](len(model_1x), color=sim.colors['cobalt'])
s_colors_3 = sim.palette_types['light'](len(model_1x), color=sim.colors['teal'  ])
e_colors_1 = sim.palette_types['light'](len(model_1ex), color=sim.colors['purple'])
e_colors_2 = sim.palette_types['light'](len(model_1ex), color=sim.colors['cobalt'])
e_colors_3 = sim.palette_types['light'](len(model_1ex), color=sim.colors['teal'  ])
v_colors_1 = dict(zip(model_1v, s_colors_1 + s_colors_1 + e_colors_1 + e_colors_1))
v_colors_2 = dict(zip(model_1v, s_colors_2 + s_colors_2 + e_colors_2 + e_colors_2))
v_colors_3 = dict(zip(model_1v, s_colors_3 + s_colors_3 + e_colors_3 + e_colors_3))
color      = {'model_1': {0: {0: v_colors_1
                              },
                          1: {0: v_colors_2
                              },
                          2: {0: v_colors_3
                              }
                          }
                }

simulation_results   = sim.integrate_models(sim_args)
figs, AX             = sim.plot_simulation_results(plot_index, 
                                                   simulation_results,
                                                   AX          = AX,
                                                   color       = color,
                                                   label       = 'state, scenario'
                                                   )

#Post processing
sim.fs(fig)
for ax in AX_:
    ax.legend(bbox_to_anchor=(0.95, 1), loc='upper left')
plt.subplots_adjust(left=0.03, right=0.92, wspace=0.22)
