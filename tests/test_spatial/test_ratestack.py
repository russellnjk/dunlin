import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
import textwrap          as tw
from collections import Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import addpath
import dunlin       as dn 
import dunlin.utils as ut
from dunlin.spatial.ratestack      import RateStack as Stack
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
#Test Rate
###############################################################################
code  = tw.dedent(stk.rate_code)
scope = {ut.diff('A') : np.zeros(4),
         'vrb0'       : np.array([1, 2, 3, 4]),
         **stk.functions
         }

# print(stk.rate_code)
exec(code, None, scope)

assert all(scope[ut.diff('A')] == [1, 2, 3, 4]) 

cax = make_colorbar_ax(AX[1])
stk.plot_rate(AX[1], 'A', scope[ut.diff('A')], cmap='coolwarm', colorbar_ax=cax)
AX[1].set_title('Rate A')

###############################################################################
#Test RHS
###############################################################################
code = stk.rhs_code

with open('output_rhs.txt', 'w') as file:
    file.write(code)

stk.numba = False

time       = 0
states     = np.arange(0, 32)
parameters = spatial_data.parameters.df.loc[0].values

d_states = stk.rhs(time, states, parameters)

fig, AX = make_fig(AX)

dxidx = stk.state2dxidx

start, stop = dxidx['A']
cax = make_colorbar_ax(AX[4])
stk.plot_rate(AX[4], 'A', d_states[start: stop], cmap='coolwarm', colorbar_ax=cax)
AX[4].set_title(ut.diff('A'))

start, stop = dxidx['B']
cax = make_colorbar_ax(AX[5])
stk.plot_rate(AX[5], 'B', d_states[start: stop], cmap='coolwarm', colorbar_ax=cax)
AX[5].set_title(ut.diff('B'))

start, stop = dxidx['C']
cax = make_colorbar_ax(AX[6])
stk.plot_rate(AX[6], 'C', d_states[start: stop], cmap='coolwarm', colorbar_ax=cax)
AX[6].set_title(ut.diff('C'))

start, stop = dxidx['D']
cax = make_colorbar_ax(AX[7])
stk.plot_rate(AX[7], 'D', d_states[start: stop], cmap='coolwarm', colorbar_ax=cax)
AX[7].set_title(ut.diff('D'))

###############################################################################
#Test RHS dct
###############################################################################
code = stk.rhsdct_code

with open('output_rhsdct.txt', 'w') as file:
    file.write(code)


time       = 0
states     = np.arange(0, 32)
parameters = spatial_data.parameters.df.loc[0].values

stk.numba = False
dct        = stk.rhsdct(time, states, parameters)

fig, AX = make_fig(AX)

cax = make_colorbar_ax(AX[8])
stk.plot_rate(AX[8], 'A', dct[ut.diff('A')], cmap='coolwarm', colorbar_ax=cax)
AX[8].set_title(ut.diff('A'))

cax = make_colorbar_ax(AX[9])
stk.plot_rate(AX[9], 'B', dct[ut.diff('B')], cmap='coolwarm', colorbar_ax=cax)
AX[9].set_title(ut.diff('B'))

cax = make_colorbar_ax(AX[10])
stk.plot_rate(AX[10], 'C', dct[ut.diff('C')], cmap='coolwarm', colorbar_ax=cax)
AX[10].set_title(ut.diff('C'))

cax = make_colorbar_ax(AX[11])
stk.plot_rate(AX[11], 'D', dct[ut.diff('D')], cmap='coolwarm', colorbar_ax=cax)
AX[11].set_title(ut.diff('D'))
