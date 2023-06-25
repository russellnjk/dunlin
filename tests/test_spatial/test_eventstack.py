import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
import textwrap          as tw
from collections import Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import addpath
import dunlin         as dn 
import dunlin.ode.ivp as ivp
import dunlin.utils   as ut
from dunlin.spatial.eventstack     import EventStack as Stack
from dunlin.datastructures.spatial import SpatialModelData
from test_spatial_data             import all_data

#Set up
plt.close('all')
plt.ion()

spatial_data = SpatialModelData.from_all_data(all_data, 'M0')

def make_fig(AX):
    span = -1, 5
    fig  = plt.figure(figsize=(10, 10))
    
    for i in range(4):
        ax  = fig.add_subplot(2, 2, i+1)#, projection='3d')
        ax.set_box_aspect(1)
        
        ax.set_xlim(*span)
        ax.set_ylim(*span)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        plt.grid(True)
        
        AX.append(ax)
    
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    return fig, AX

def make_colorbar_ax(ax):
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="5%", pad=0.05)
    return cax

AX = []

###############################################################################
#Test Instantiation
###############################################################################
stk = Stack(spatial_data)

fig, AX = make_fig(AX)

domain_type_args = {'facecolor': {'cytosolic'     : 'steel',
                                  'extracellular' : 'salmon'
                                  }
                    }

stk.plot_voxels(AX[0], domain_type_args=domain_type_args)

###############################################################################
#Test Trigger
###############################################################################
time       = 0
states     = stk.reshape(np.arange(0, 4))
parameters = spatial_data.parameters.df.loc[0].values

trigger_func = stk._events[0].trigger_func
residual     = trigger_func(time, states, parameters)

assert residual == -2

###############################################################################
#Test Assignment
###############################################################################
time       = 0 
states     = stk.reshape(np.arange(0, 4))
parameters = spatial_data.parameters.df.loc[0].values

assign_func = stk._events[0]._execute
new_y, new_p = assign_func(time, states, parameters)

