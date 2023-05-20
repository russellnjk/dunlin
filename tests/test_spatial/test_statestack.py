import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
import textwrap          as tw

import addpath
import dunlin       as dn 
import dunlin.utils as ut
from dunlin.spatial.statestack     import StateStack as Stack
from dunlin.datastructures.spatial import SpatialModelData
from test_spatial_data             import all_data

#Set up
plt.close('all')
plt.ion()


spatial_data = SpatialModelData.from_all_data(all_data, 'M0')

span = -1, 5
fig  = plt.figure(figsize=(15, 10))
AX   = []
for i in range(6):
    ax  = fig.add_subplot(2, 3, i+1)#, projection='3d')
    ax.set_box_aspect(1)
    ax.set_box_aspect()
    ax.set_xlim(*span)
    ax.set_ylim(*span)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.grid(True)
    
    AX.append(ax)

###############################################################################
#Test Instantiation
###############################################################################
stk = Stack(spatial_data)

domain_type_args = {'facecolor': {'cytosolic'     : 'steel',
                                  'extracellular' : 'salmon'
                                  }
                    }

stk.plot_voxels(AX[0], domain_type_args=domain_type_args)

assert len(stk.element2idx) == 32
assert len(stk.state2dxidx) == 4 

###############################################################################
#Test State Code
###############################################################################
states = list(range(0, 32))
code   = tw.dedent(stk.state_code)
scope  = {}
exec(code, None, scope)

print(stk.state_code)
# print(scope)

assert scope == {'A': [0, 1, 2, 3],
                 'B': [4, 5, 6, 7],
                 'C': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                 'D': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
                 }

cmap_and_values = {'cmap'   : 'coolwarm',
                   'values' : [0, 31]
                   }

norms, color_func = stk.make_scaled_cmap(A=cmap_and_values, D=cmap_and_values)

state_args = {'facecolor': color_func,
              }

stk.plot_state(AX[1], 'A', scope['A'], state_args)
AX[1].set_title('A')

stk.plot_state(AX[2], 'D', scope['D'], state_args)
AX[2].set_title('D')


AX[3].grid(False)
cb = mpl.colorbar.Colorbar(ax=AX[3], cmap='coolwarm', norm=norms['A'])
AX[3].set_title('state colormap')


code  = tw.dedent(stk.diff_code)
scope = {}
exec(code, {'__zeros': np.zeros, '__float64': np.float64}, scope)

print(stk.diff_code)
# print(scope)

assert all( scope[ut.diff('A')] == np.zeros(4 ) )
assert all( scope[ut.diff('B')] == np.zeros(4 ) )
assert all( scope[ut.diff('C')] == np.zeros(12) )
assert all( scope[ut.diff('D')] == np.zeros(12) )

stk.plot_diff(AX[4], 'A', scope[ut.diff('A')], state_args)
AX[4].set_title(ut.diff('A'))

# stk.plot_state(AX[5], ut.diff('D'), scope[ut.diff('D')], state_args)
# AX[5].set_title(ut.diff('D'))

###############################################################################
#Test Parameter Code
###############################################################################
parameters = list(range(0, 11))
code       = tw.dedent(stk.parameter_code)
scope      = {}
exec(code, None, scope)

print(stk.parameter_code)
# print(scope)

assert scope == {'k_degA': 0, 
                 'k_synB': 1, 
                 'k_synD': 2, 
                 'k_degD': 3, 
                 'k_pump': 4, 
                 'J_B_x': 5, 
                 'J_B_y': 6, 
                 'J_C': 7, 
                 'F_B_x': 8, 
                 'F_B_y': 9, 
                 'F_C': 10
                 }

###############################################################################
#Test Function Code
###############################################################################
code       = tw.dedent(stk.function_code)
scope      = {}
exec(code, None, scope)

print(stk.function_code)
assert scope['func0'](2, 3) == -6
