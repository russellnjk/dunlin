import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
import textwrap          as tw
from collections             import Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba                   import njit

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

template  = '''
{body}\n\treturn {return_val}
    
'''

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

a = stk.surface_data['cytosolic', 'extracellular']['mappings']

###############################################################################
#Test Variable Code
###############################################################################
code  = (stk.rhsdef           + '\n' 
         + stk.state_code     + '\n' 
         + stk.diff_code      + '\n'
         + stk.parameter_code + '\n' 
         + stk.function_code  + '\n' 
         + stk.variable_code
         )
code  = template.format(body       = code,
                        return_val = 'vrb0, vrb1, vrb2, vrb3'
                        )
print(code)

scope = {}
exec(code, {**stk.rhs_functions}, scope)
# print(scope.keys())

model      = njit(scope['model_M0'])
time       = 0
states     = np.array([4, 4, 4, 4,
                        0, 1, 2, 3,
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                        ])
parameters = np.array([3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0])

r = model(time, states, parameters, **stk.args)

assert all(r[0] == -12)
assert Counter(r[1]) == {0: 2, 4: 1, 10: 1, 24: 1, 36: 1, 42: 1, 60: 1}
assert r[2]     == 1
assert Counter(r[3]) == {0: 2, -48: 1, -120: 1, -288: 1, -432: 1, -504: 1, -720: 1}

###############################################################################
#Test Reaction Code
###############################################################################
code  = (stk.rhsdef           + '\n' 
         + stk.state_code     + '\n' 
         + stk.diff_code      + '\n'
         + stk.parameter_code + '\n' 
         + stk.function_code  + '\n' 
         + stk.variable_code  + '\n'
         + stk.reaction_code
         )
rv    = 'synB, synD, pumpB, k_synD, vrb2, C, D, _d_D, _d_C, _d_B'
code  = template.format(body       = code,
                        return_val = rv
                        )
print(code)

scope = {}
exec(code, {**stk.rhs_functions}, scope)
# print(scope.keys())

model      = njit(scope['model_M0'])
time       = 0
states     = np.array([4, 4, 4, 4,
                        0, 1, 2, 3,
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                        ])
parameters = spatial_data.parameters.df.loc[0]
parameters = np.array([3, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0])

r = model(time, states, parameters, **stk.args)

#Scalar
assert r[0] == 1

#Bulk reaction
k_synD, vrb2, C, D = r[3:7]
assert all(r[1] == k_synD*C - vrb2*D)

_d_D  = r[7]
assert all(_d_D == -vrb2*D) #k_synD*C is zero

#Surface reaction
assert Counter(r[2]) == {0: 2, 2: 2, 4: 2, 6: 2}

pumpB = r[2]
_d_C  = r[8]
temp  = np.zeros(12)

temp[[1, 4, 2, 5, 6, 9, 7, 10]] = pumpB

assert all(_d_C == vrb2*D + temp)

#Plot
cax = make_colorbar_ax(AX[1])
stk.plot_reaction(AX[1], 'pumpB', pumpB, cmap='coolwarm', colorbar_ax=cax)
AX[1].set_title('Surface reaction pumpB')

cax = make_colorbar_ax(AX[2])
stk.plot_diff(AX[2], 'C', _d_C, cmap='coolwarm', colorbar_ax=cax)
AX[2].set_title(ut.diff('B'))

_d_B = r[9]
cax = make_colorbar_ax(AX[3])
stk.plot_diff(AX[3], 'B', _d_B, cmap='PiYG', colorbar_ax=cax)
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
stk.plot_reaction(AX[4], 'synD', r[1], cmap='coolwarm', colorbar_ax=cax)
AX[4].set_title('Bulk reaction synD')

cax = make_colorbar_ax(AX[5])
stk.plot_diff(AX[5], 'D', r[6], cmap='coolwarm', colorbar_ax=cax)
AX[5].set_title(ut.diff('D'))

###############################################################################
#Test Performance 
###############################################################################
print('Testing M1')
spatial_data = SpatialModelData.from_all_data(all_data, 'M1')

print('Init M1')
stk_ = Stack(spatial_data)

code  = (stk_.rhsdef          + '\n' 
          + stk_.state_code     + '\n' 
          + stk_.diff_code      + '\n'
          + stk_.parameter_code + '\n' 
          + stk_.function_code  + '\n' 
          + stk_.variable_code  + '\n'
          + stk_.reaction_code
          )
rv    = 'synB, synD, pumpB, k_synD, vrb2, C, D, _d_D, _d_C, _d_B'

code  = template.format(body       = code,
                        return_val = rv
                        )
# print(code)

scope = {}
exec(code, {**stk.rhs_functions}, scope)
# print(scope.keys())

model_      = njit(scope['model_M1'])
time_       = 0
states_     = stk_.expand_init(np.array([1, 2, 3, 4]))
parameters_ = np.array([3, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0])

print('Calling M1')
r = model_(time_, states_, parameters_, **stk_.args)

#M2
spatial_data = SpatialModelData.from_all_data(all_data, 'M2')
print('Init M2')
stk__ = Stack(spatial_data)

code  = (stk__.rhsdef            + '\n' 
          + stk__.state_code     + '\n' 
          + stk__.diff_code      + '\n'
          + stk__.parameter_code + '\n' 
          + stk__.function_code  + '\n' 
          + stk__.variable_code  + '\n'
          + stk__.reaction_code
          )
rv    = 'synB, synD, pumpB, k_synD, vrb2, C, D, _d_D, _d_C, _d_B'

code  = template.format(body       = code,
                        return_val = rv
                        )
# print(code)

scope = {}
exec(code, {**stk.rhs_functions}, scope)
# print(scope.keys())

model__      = njit(scope['model_M2'])
time__       = 0
states__     = stk__.expand_init(np.array([1, 2, 3, 4]))
parameters__ = np.array([3, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0])

print('Calling M2')
r = model__(time__, states__, parameters__, **stk__.args)

###############################################################################
#Test RHSDct
###############################################################################
spatial_data = SpatialModelData.from_all_data(all_data, 'M0')
stk          = Stack(spatial_data)

code  = (stk.rhsdef           + '\n' 
         + stk.state_code     + '\n' 
         + stk.diff_code      + '\n'
         + stk.parameter_code + '\n' 
         + stk.function_code  + '\n' 
         + stk.variable_code  + '\n'
         + stk.reaction_code
         )
code  = template.format(body       = code,
                        return_val = 'A, D, synB, pumpB, _d_A, _d_D'
                        )
print(code)

scope = {}
exec(code, {**stk.rhs_functions}, scope)

model      = njit(scope['model_M0'])
time       = np.array([0, 1, 2])
states     = np.array([4, 4, 4, 4,
                        0, 1, 2, 3,
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                        ])
states     = np.array([states, states+1, states+2]).T
parameters = np.array([3, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0])
parameters = np.array([parameters, parameters, parameters]).T

r = model(time, states, parameters, **stk.args)