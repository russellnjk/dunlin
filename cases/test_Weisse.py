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

#Read files
filename = 'Weisse1.dun'
dun_data, models = dn.read_file(filename)
model            = models['Weisse'] 

#Integrate
sim_results = sim.integrate_model(model, 
                                  multiply=True, 
                                  overlap=True, 
                                  include_events=True
                                  )

#Plot
#This will override line_args in the .dun file (if applicable)
line_args   = {'label': 'estimate', 
               }

#Make figure and axes
fig0, AX0 = dn.figure(3, 4)
fig1, AX1 = dn.figure(3, 4, 11)
AX        = AX0 + AX1

states      = model.get_state_names() 
AX_         = dict(zip(states, AX))

sim.plot_sim_results(sim_results, AX_, **line_args)
for x, ax in zip(states, AX):
    ax.set_title(x)
    dn.scilimit(ax)
    ymax = ax.yaxis.get_data_interval()[1]
    ymax = max(1e-3, ymax)
    ax.set_ylim(0, ymax)
 