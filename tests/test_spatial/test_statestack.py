import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
import textwrap          as tw
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def make_fig(AX):
    span = -1, 5
    fig  = plt.figure(figsize=(10, 10))
    
    for i in range(4):
        ax  = fig.add_subplot(2, 2  , i+1)#, projection='3d')
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

assert len(stk.element2idx) == 32
assert len(stk.state2dxidx) == 4 

###############################################################################
#Test State Code
###############################################################################
print(stk.state_code)

states = list(range(0, 32))
code   = tw.dedent(stk.state_code)
scope  = stk.functions
exec(code, {'states': states}, scope)
# print(scope)

assert scope['A'] == [0, 1, 2, 3]
assert scope['B'] == [4, 5, 6, 7]
assert scope['C'] == [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
assert scope['D'] == [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

cax = make_colorbar_ax(AX[1])
stk.plot_state(AX[1], 'A', scope['A'], cmap='coolwarm', colorbar_ax=cax)
AX[1].set_title('A')

cax = make_colorbar_ax(AX[2])
stk.plot_state(AX[2], 'D', scope['D'], cmap='coolwarm', colorbar_ax=cax)
AX[2].set_title('D')

code  = tw.dedent(stk.diff_code)
scope = stk.functions
exec(code, None, scope)

print(stk.diff_code)
# print(scope)

assert all( scope[ut.diff('A')] == np.zeros(4 ) )
assert all( scope[ut.diff('B')] == np.zeros(4 ) )
assert all( scope[ut.diff('C')] == np.zeros(12) )
assert all( scope[ut.diff('D')] == np.zeros(12) )

stk.plot_diff(AX[3], 'A', scope[ut.diff('A')], cmap='coolwarm')
AX[3].set_title(ut.diff('A'))

###############################################################################
#Test Parameter Code
###############################################################################
parameters = list(range(0, 11))
code       = tw.dedent(stk.parameter_code)
scope      = stk.functions
exec(code, None, scope)

print(stk.parameter_code)
# print(scope)

assert scope['k_degA'] == 0
assert scope['k_synB'] == 1 
assert scope['k_synD'] == 2 
assert scope['k_degD'] == 3
assert scope['k_pump'] == 4
assert scope['J_B'   ] == 5
assert scope['J_C_x' ] == 6
assert scope['J_C_y' ] == 7
assert scope['F_B'   ] == 8 
assert scope['F_C_x' ] == 9 
assert scope['F_C_y' ] == 10 

###############################################################################
#Test Function Code
###############################################################################
code       = tw.dedent(stk.function_code)
scope      = {}
exec(code, None, scope)

print(stk.function_code)
assert scope['func0'](2, 3) == -6

