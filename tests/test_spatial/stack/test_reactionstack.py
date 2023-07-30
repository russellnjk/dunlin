import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
import textwrap          as tw
from collections import Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import addpath
import dunlin       as dn 
import dunlin.utils as ut
from dunlin.spatial.stack.reactionstack  import ReactionStack as Stack
from dunlin.datastructures.spatial       import SpatialModelData
from test_spatial_data                   import all_data

#Set up
plt.close('all')
plt.ion()

spatial_data = SpatialModelData.from_all_data(all_data, 'M0')

def make_fig(AX):
    span = -1, 5
    fig  = plt.figure(figsize=(15, 10))
    
    for i in range(6):
        ax  = fig.add_subplot(2, 3, i+1)#, projection='3d')
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
#Test Variable Code
###############################################################################
code  = tw.dedent(stk.variable_code)
scope = {'func0'  : lambda a, b: a*b,
         'k_pump' : 2,
         'k_degA' : 3,
         'A'      : np.ones(4),
         'B'      : np.array(list(range(0, 4))),
         'C'      : np.array(list(range(0, 12))),
          **stk.rhs_functions
          }
exec(code, {}, scope)
print(stk.variable_code)
# print(scope)

assert all(scope['vrb0'] == 3)
assert Counter(scope['vrb1']) == {0: 2, 4: 1, 10: 1, 24: 1, 36: 1, 42: 1, 60: 1}
assert scope['vrb2']     == 1
assert Counter(scope['vrb3']) == {0: 2, 12: 1, 30: 1, 72: 1, 108: 1, 126: 1, 180: 1}

###############################################################################
#Test Reaction Code
###############################################################################
print(stk.reaction_code)

code  = tw.dedent(stk.reaction_code)
scope = {'k_synB'     : 1,
          'k_pump'     : 2,
          'k_synD'     : 3,
          'vrb2'       : 0,
          'A'          : np.arange(4),
          'B'          : np.arange(4),
          'C'          : np.arange(12),
          'D'          : np.arange(12),
          ut.diff('A') : np.zeros(4),
          ut.diff('B') : np.zeros(4),
          ut.diff('C') : np.zeros(12),
          ut.diff('D') : np.zeros(12),
          **stk.rhs_functions
          }
exec(code, None, scope)
# print(scope)

#Scalar
assert scope['synB'] == 1

#Bulk reaction with vector
a = scope['k_synD']*scope['C'] - scope['vrb2']*scope['D']
assert all(a == scope['synD'])

#Surface reaction with vector
assert Counter(scope['pumpB']) == {0: 2, 2: 2, 4: 2, 6: 2}

#Plot
cax = make_colorbar_ax(AX[1])
stk.plot_reaction(AX[1], 'pumpB', scope['pumpB'], cmap='coolwarm', colorbar_ax=cax)
AX[1].set_title('Surface reaction pumpB')

cax = make_colorbar_ax(AX[2])
stk.plot_diff(AX[2], 'C', scope[ut.diff('C')], cmap='coolwarm', colorbar_ax=cax)
AX[2].set_title(ut.diff('B'))

cax = make_colorbar_ax(AX[3])
stk.plot_diff(AX[3], 'B', scope[ut.diff('B')], cmap='PiYG', colorbar_ax=cax)
AX[3].set_title(ut.diff('C'))

#Bulk reaction
scope = {'k_synB'     : 0,
          'k_pump'     : 0,
          'k_synD'     : 2,
          'vrb1'       : 0,
          'vrb2'       : 0,
          'A'          : np.arange(4),
          'B'          : np.arange(4),
          'C'          : np.arange(12),
          'D'          : np.arange(12),
          ut.diff('A') : np.zeros(4),
          ut.diff('B') : np.zeros(4),
          ut.diff('C') : np.zeros(12),
          ut.diff('D') : np.zeros(12),
          **stk.rhs_functions
          }
exec(code, None, scope)

cax = make_colorbar_ax(AX[4])
stk.plot_reaction(AX[4], 'synD', scope['synD'], cmap='coolwarm', colorbar_ax=cax)
AX[4].set_title('Bulk reaction synD')

cax = make_colorbar_ax(AX[5])
stk.plot_diff(AX[5], 'D', scope[ut.diff('D')], cmap='coolwarm', colorbar_ax=cax)
AX[5].set_title(ut.diff('D'))


