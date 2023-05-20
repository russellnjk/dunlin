import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
import textwrap          as tw
from collections import Counter

import addpath
import dunlin       as dn 
import dunlin.utils as ut
from dunlin.spatial.reactionstack  import ReactionStack as Stack
from dunlin.datastructures.spatial import SpatialModelData
from test_spatial_data             import all_data

#Set up
plt.close('all')
plt.ion()

global_scope = {'__array': np.array, '__zeros': np.zeros}

advection    = all_data['M0'].pop('advection')
diffusion    = all_data['M0'].pop('diffusion') 
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
          }
exec(code, global_scope, scope)
print(stk.variable_code)
# print(scope)

assert all(scope['vrb0'] == 3)
assert Counter(scope['vrb1']) == {0: 2, 4: 1, 10: 1, 24: 1, 36: 1, 42: 1, 60: 1}
assert scope['vrb2']     == 1
assert Counter(scope['vrb3']) == {0: 2, 12: 1, 30: 1, 72: 1, 108: 1, 126: 1, 180: 1}

vrb = scope['vrb1']
B   = scope['B']
C   = scope['C']
k   = scope['k_pump']

iB  = stk.get_bulk_idx([1.5, 2.5])
iC  = stk.get_bulk_idx([0.5, 2.5])
iS  = stk.get_surface_idx([1, 2.5])

assert vrb[iS] == k*B[iB]*C[iC]

iB  = stk.get_bulk_idx([2.5, 2.5])
iC  = stk.get_bulk_idx([3.5, 2.5])
iS  = stk.get_surface_idx([3, 2.5])

assert vrb[iS] == k*B[iB]*C[iC]

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
         }
exec(code, global_scope, scope)
# print(scope)

#Scalar
assert scope['synB'] == 1

#Bulk reaction with vector
a = scope['k_synD']*scope['C'] - scope['vrb2']*scope['D']
assert all(a == scope['synD'])

#Surface reaction with vector
assert Counter(scope['pumpB']) == {0: 2, 2: 2, 4: 2, 6: 2}

rxn = scope['pumpB']
B   = scope['B']
k   = scope['k_pump']

iB  = stk.get_bulk_idx([1.5, 2.5])
iS  = stk.get_surface_idx([1, 2.5])

assert rxn[iS] == k*B[iB]

scope = {'k_synB'     : 0,
         'k_pump'     : 2,
         'k_synD'     : 0,
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
         }
exec(code, global_scope, scope)

surface = ('cytosolic', 'extracellular')
idxs    = stk.get_surface(surface, 'extracellular')
iC      = list(idxs.values())
iR      = list(idxs.keys())
rxn     = scope['pumpB']

dC     = np.zeros(12)
dC[iC] = rxn[iR]

assert Counter({0.0: 6, 2.0: 2, 4.0: 2, 6.0: 2}) == {0.0: 6, 2.0: 2, 4.0: 2, 6.0: 2} 

#Plot
cmap_and_values = {'cmap'   : 'coolwarm',
                   'values' : [0, 6]
                   }

color_func = stk.make_scaled_cmap(_default=cmap_and_values)

reaction_args = {'color': color_func,
                 }
stk.plot_reaction(AX[1], 'pumpB', scope['pumpB'], reaction_args)
AX[1].set_title('pumpB')

AX[2].grid(False)
cb = mpl.colorbar.Colorbar(ax   = AX[2], 
                           cmap = color_func.cmaps['_default'], 
                           norm = color_func.norms['_default']
                           )
AX[2].set_title('pumpB colormap')

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
         }
exec(code, global_scope, scope)

cmap_and_values = {'cmap'   : 'coolwarm',
                   'values' : scope['synD']
                   }

color_func = stk.make_scaled_cmap(_default=cmap_and_values)

reaction_args = {'facecolor': color_func,
                 }

stk.plot_reaction(AX[3], 'synD', scope['synD'], reaction_args)
AX[3].set_title('synD')

AX[4].grid(False)
cb = mpl.colorbar.Colorbar(ax   = AX[4], 
                           cmap = color_func.cmaps['_default'], 
                           norm = color_func.norms['_default']
                           )
AX[4].set_title('synD colormap')



diff_args = {'facecolor': color_func
             }
stk.plot_diff(AX[5], 'D', scope[ut.diff('D')], diff_args)
AX[5].set_title(ut.diff('D'))
