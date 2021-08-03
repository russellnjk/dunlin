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

#Read the file
filename         = 'Turing1.dun'
dun_data, models = dn.read_file(filename)
model            = models['Turing'] 

#Integrate
sim_results = sim.integrate_model(model, 
                                  multiply=True, 
                                  overlap=True, 
                                  include_events=True
                                  )

#Plot
#This will override line_args in the .dun file (if applicable)
line_args   = {'label': 'state, scenario', 
               }

#Make figure and axes
fig0, AX0 = dn.figure(1, 2)
fig1, AX1 = dn.figure(2, 4)
AX        = AX0 + AX1

states      = models['M1'].get_state_names() 
x_states    = states[:5]
y_states    = states[5:]
AX_x_states = {x: AX[0] for x in x_states}
AX_y_states = {x: AX[1] for x in y_states}
AX_         = {**AX_x_states, 
               **AX_y_states, 
               **dict(zip(model.exvs, AX[2:]))
               }

sim.plot_sim_results(sim_results, AX_, **line_args)

#Post-processing e.g. legend, scientific notation
[ax.legend() for ax in AX]
dn.scilimit(AX, lb=-2, ub=3, nbins=4)

#Alternative layouts
fig, AX = dn.gridspec(4, 4,
                      [0, 2, 0, 2],
                      [2, 4, 0, 2],
                      [0, 1, 2, 3],
                      [1, 2, 2, 3],
                      [2, 3, 2, 3],
                      [3, 4, 2, 3],
                      [0, 1, 3, 4],
                      [1, 2, 3, 4],
                      [2, 3, 3, 4],
                      [3, 4, 3, 4],
                      )

states      = model.get_state_names() 
x_states    = states[:5]
y_states    = states[5:]
AX_x_states = {x: AX[0] for x in x_states}
AX_y_states = {x: AX[1] for x in y_states}
AX_         = {**AX_x_states, 
               **AX_y_states, 
               **dict(zip(model.exvs, AX[2:]))
               }

#Post-processing e.g. legend, scientific notation
sim.plot_sim_results(sim_results, AX_, **line_args)
[ax.legend() for ax in AX]
dn.scilimit(AX, lb=-2, ub=3, nbins=4)
