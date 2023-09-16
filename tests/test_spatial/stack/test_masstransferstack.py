import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
import numpy.ma          as ma
import textwrap          as tw
from collections             import Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba                   import njit  

import addpath
import dunlin       as dn 
import dunlin.utils as ut
from dunlin.spatial.stack.masstransferstack import (calculate_advection,
                                                    calculate_diffusion,
                                                    calculate_neumann,
                                                    calculate_dirichlet,
                                                    )
from dunlin.spatial.stack.masstransferstack import MassTransferStack as Stack
from dunlin.datastructures.spatial          import SpatialModelData
from test_spatial_data                      import all_data



#Set up
plt.close('all')
plt.ion()

###############################################################################
#Test Advection Calculator
###############################################################################
X     = np.array([0, 10, 20, 30, 40, 50])
coeff = -1

left_srcs    = [1, 2, 3, 4]
left_dsts    = [0, 1, 2, 3]
left_scales  = [1., 1., 1., 1.]

right_srcs    = [ 0, 1, 2, 3]
right_dsts    = [ 1, 2, 3, 4]
right_scales  = [ 1., 1., 1., 1.]

dtype = [('source', np.int32), ('destination', np.int32), ('area', np.float64)]

left_mapping  = np.array([*zip(left_srcs, left_dsts, left_scales)],    dtype=dtype)
right_mapping = np.array([*zip(right_srcs, right_dsts, right_scales)], dtype=dtype)
volumes       = np.ones(len(X))

print('Test advection calculation')
dX = calculate_advection(X, coeff, left_mapping, right_mapping, volumes)
print(dX)

assert all(dX == [10, 10, 10, 10, -40, 0])

coeff = 1
dX    = calculate_advection(X, coeff, left_mapping, right_mapping, volumes)
print(dX)
assert all(dX == [0, -10, -10, -10, 30, 0])

#Scale up
print('Scaling up')
n_tiles = 20000

def tile(arr, n, m):
    result = []
    for i in range(n):
        for ii, v in enumerate(arr):
            v += m*i
            result.append(v)
    
    return result

left_srcs   = tile(left_srcs, n_tiles, len(X))
left_dsts   = tile(left_dsts, n_tiles, len(X))
left_scales = left_scales*n_tiles

right_srcs   = tile(right_srcs, n_tiles, len(X))
right_dsts   = tile(right_dsts, n_tiles, len(X))
right_scales = right_scales*n_tiles

X     = np.tile(X, n_tiles)
coeff = 1

left_mapping  = np.array([*zip(left_srcs, left_dsts, left_scales)],    dtype=dtype)
right_mapping = np.array([*zip(right_srcs, right_dsts, right_scales)], dtype=dtype)
volumes       = np.ones(len(X))

dX = calculate_advection(X, coeff, left_mapping, right_mapping, volumes)
print(dX)

###############################################################################
#Set Up Stack Tests
###############################################################################
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

###############################################################################
#Test Advection Code
###############################################################################
left_mapping  = stk.args['_adv_cytosolic_1_left']
right_mapping = stk.args['_adv_cytosolic_1_right']

r = calculate_advection(np.array([4, 4, 4, 4]), 2, left_mapping, right_mapping, volumes)

assert Counter(r) == {-8: 2, 8: 2}

left_mapping  = stk.args['_adv_cytosolic_2_left']
right_mapping = stk.args['_adv_cytosolic_2_right']

r = calculate_advection(np.array([4, 4, 4, 4]), 2, left_mapping, right_mapping, volumes)

assert Counter(r) == {-8: 2, 8: 2}

code  = (stk.rhsdef           + '\n' 
         + stk.state_code     + '\n' 
         + stk.diff_code      + '\n'
         + stk.parameter_code + '\n' 
         + stk.advection_code
         )
code  = template.format(body       = code,
                        return_val = 'A, B, C, D, _adv_B, _adv_C, _adv_D, _d_A, _d_B, _d_C, _d_D'
                        )

# print(code)

scope = {}
exec(code, {**stk.rhs_functions}, scope)
# print(scope.keys())

model      = njit(scope['model_M0'])
time       = 0
states     = np.array([4, 4, 4, 4,
                       4, 4, 4, 4,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                       ])
parameters = np.array([0, 0, 0, 0, 0, 0, 0, 0, -2, 1, -1])

r = model(time, states, parameters, **stk.args)

assert Counter(r[4]) == {16: 1, -16: 1, 0: 2}

cax = make_colorbar_ax(AX[1])
stk.plot_advection(AX[1], 'B', r[4], cmap='coolwarm', colorbar_ax=cax)
AX[1].set_title(ut.adv('B'))

cax = make_colorbar_ax(AX[2])
stk.plot_advection(AX[2], 'C', r[5], cmap='coolwarm', colorbar_ax=cax)
AX[2].set_title(ut.adv('C'))

cax = make_colorbar_ax(AX[3])
stk.plot_advection(AX[3], 'D', r[6], cmap='coolwarm', colorbar_ax=cax)
AX[3].set_title(ut.adv('D'))

###############################################################################
#Test Diffusion Calculation
###############################################################################
X     = np.array([0, 10, 20, 30, 40, 50])
coeff = 1

left_srcs     = [1, 2, 3, 4]
left_dsts     = [0, 1, 2, 3]
left_scales   = [1, 1, 1, 1]
dtype         = [('source', np.int32), ('destination', np.int32), ('area/distance', np.float64)]
left_mapping  = np.array([*zip(left_srcs, left_dsts, left_scales)], dtype=dtype)
volumes       = np.ones(6)

dX = calculate_diffusion(X, coeff, left_mapping, volumes)
# print(dX)

assert all(dX == [10, 0, 0, 0, -10, 0])

###############################################################################
#Setup
###############################################################################
fig, AX = make_fig(AX)

stk.plot_voxels(AX[4], domain_type_args=domain_type_args)

###############################################################################
#Test Diffusion Code
###############################################################################
left_mapping  = stk.args['_dfn_cytosolic_1_left']

r = calculate_diffusion(np.array([4, 0, 0, 0]), 2, left_mapping, volumes)

assert Counter(r == {-8: 1, 8: 1, 0: 2})

left_mapping  = stk.args['_dfn_cytosolic_2_left']

r = calculate_diffusion(np.array([4, 0, 0, 0]), 2, left_mapping, volumes)

assert Counter(r == {-8: 1, 8: 1, 0: 2})

code  = (stk.rhsdef           + '\n' 
         + stk.state_code     + '\n' 
         + stk.diff_code      + '\n'
         + stk.parameter_code + '\n' 
         + stk.diffusion_code
         )
code  = template.format(body       = code,
                        return_val = 'A, B, C, D, _dfn_B, _dfn_C, _dfn_D, _d_A, _d_B, _d_C, _d_D'
                        )

# print(code)

scope = {}
exec(code, {**stk.rhs_functions}, scope)
# print(scope.keys())

model      = njit(scope['model_M0'])
time       = 0
states     = np.array([4, 4, 4, 4,
                       4, 0, 0, 0,
                       4, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 4,
                       4, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 4,
                       ])
parameters = np.array([0, 0, 0, 0, 0, 2, 2, 3, 0, 0, 0])

r = model(time, states, parameters, **stk.args)

assert Counter(r[4]) == {-16: 1, 8: 2, 0: 1}
assert Counter(r[5]) == {-20: 4, 8: 4, 12: 4}
assert Counter(r[6]) == {-8 : 4, 4: 8}

cax = make_colorbar_ax(AX[5])
stk.plot_advection(AX[5], 'B', r[4], cmap='coolwarm', colorbar_ax=cax)
AX[5].set_title(ut.dfn('B'))

cax = make_colorbar_ax(AX[6])
stk.plot_advection(AX[6], 'C', r[5], cmap='coolwarm', colorbar_ax=cax)
AX[6].set_title(ut.dfn('C'))

cax = make_colorbar_ax(AX[7])
stk.plot_advection(AX[7], 'D', r[6], cmap='coolwarm', colorbar_ax=cax)
AX[7].set_title(ut.dfn('D'))

###############################################################################
#Test Boundary Calculation
###############################################################################
#Neumann
X       = np.array([0,  10, 20, 30, 40, 50])
fluxes  = np.array([5,   5,  5,  5,  5,  5])

srcs    = [1, 3]
scales  = [1, 1]
dtype   = [('source', np.int32), ('area', np.float64)]
mapping = np.array([*zip(srcs, scales)], dtype=dtype)
volumes = np.ones(6)

dX = calculate_neumann(X, fluxes, mapping, volumes)

assert all(dX == [0, 5, 0, 5, 0, 0])

#Dirichlet
X              = np.array([0,  10, 20, 30, 40, 50])
concentrations = np.array([5,   5,  5,  5,  5,  5])
coeff          = 2

srcs    = [1, 3]
scales  = [1, 1]
dtype   = [('source', np.int32), ('area/distance', np.float64)]
mapping = np.array([*zip(srcs, scales)], dtype=dtype)
volumes = np.ones(6)

dX = calculate_dirichlet(X, coeff, concentrations, mapping, volumes)

assert all(dX == [0, -10, 0, -50, 0, 0])

##############################################################################
#Setup
##############################################################################

#Plot
fig, AX = make_fig(AX)

stk.plot_voxels(AX[8], domain_type_args=domain_type_args)

###############################################################################
#Test Boundary Condition Code
###############################################################################
left_mapping  = stk.args['_bc_extracellular_1_min']
volumes       = np.ones(12)

X = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
v = np.ones(12)*2
r = calculate_neumann(X, v, left_mapping, volumes)

assert all(r == np.array([2, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 0]))

right_mapping = stk.args['_bc_extracellular_1_max']
volumes       = np.ones(12)

X = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
v = np.ones(12)*2
r = calculate_neumann(X, v, right_mapping, volumes)

assert all(r == np.array([0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 0, 2]))

code  = (stk.rhsdef           + '\n' 
         + stk.state_code     + '\n' 
         + stk.diff_code      + '\n'
         + stk.parameter_code + '\n' 
         + stk.boundary_condition_code
         )
code  = template.format(body       = code,
                        return_val = 'A, B, C, D, _bc_C, _d_C'
                        )

# print(code)

scope = {}
exec(code, {**stk.rhs_functions}, scope)
# print(scope.keys())

model      = njit(scope['model_M0'])
time       = 0
states     = np.array([0, 0, 0, 0,
                       0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       ])
parameters = np.array([0, 0, 0, 0, 0, 2, 2, 3, 0, 0, 0])

r = model(time, states, parameters, **stk.args)

assert all(r[4] == np.array([ 1, 0, 0, -1, 1, -1, 1, -1, 1, 0, 0, -1]))

cax = make_colorbar_ax(AX[9])
stk.plot_boundary_condition(AX[9], 'C', r[4], cmap='coolwarm', colorbar_ax=cax)
AX[9].set_title(ut.bc('C') + ' Neumann only')

states     = np.array([0, 0, 0, 0,
                       0, 0, 0, 0,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       ])

r = model(time, states, parameters, **stk.args)

assert all(r[4] == np.array([-2, -3, -3, -4,  1, -1,  1, -1, -2, -3, -3, -4]))

cax = make_colorbar_ax(AX[10])
stk.plot_boundary_condition(AX[10], 'C', r[4], cmap='coolwarm', colorbar_ax=cax)
AX[10].set_title(ut.bc('C'))

###############################################################################
#Test Performance
###############################################################################
#M0
print('Testing M0')
spatial_data = SpatialModelData.from_all_data(all_data, 'M0')

print('Init M0')
stk = Stack(spatial_data)

code  = (stk.rhsdef           + '\n' 
          + stk.state_code     + '\n' 
          + stk.diff_code      + '\n'
          + stk.parameter_code + '\n' 
          + stk.advection_code + '\n'
          + stk.diffusion_code +'\n'
          + stk.boundary_condition_code
          )
rv    = 'A, B, C, D, _d_A, _d_D, _d_C, _d_B'

code  = template.format(body       = code,
                        return_val = rv
                        )
# print(code)

scope = {}
exec(code, {**stk.rhs_functions}, scope)
# print(scope.keys())

model      = njit(scope['model_M0'])
time       = 0
states     = stk.expand_init(np.array([1, 2, 3, 4]))
parameters = np.array([0, 0, 0, 0, 0, 2, 2, 3, 1, 1, 1])

r = model(time, states, parameters, **stk.args)

#M1
print('Testing M1')
spatial_data = SpatialModelData.from_all_data(all_data, 'M1')

print('Init M1')
stk_ = Stack(spatial_data)

code  = (stk_.rhsdef           + '\n' 
          + stk_.state_code     + '\n' 
          + stk_.diff_code      + '\n'
          + stk_.parameter_code + '\n' 
          + stk_.advection_code + '\n'
          + stk_.diffusion_code +'\n'
          + stk_.boundary_condition_code
          )
rv    = 'A, B, C, D, _d_A, _d_D, _d_C, _d_B'

code  = template.format(body       = code,
                        return_val = rv
                        )
# print(code)

scope = {}
exec(code, {**stk_.rhs_functions}, scope)
# print(scope.keys())

model_      = njit(scope['model_M1'])
time_       = 0
states_     = stk_.expand_init(np.array([1, 2, 3, 4]))
parameters_ = np.array([0, 0, 0, 0, 0, 2, 2, 3, 1, 1, 1])

r = model_(time_, states_, parameters_, **stk_.args)

#M2
print('Testing M2')
spatial_data = SpatialModelData.from_all_data(all_data, 'M2')
print('Init M2')
stk__ = Stack(spatial_data)

code  = (stk__.rhsdef            + '\n' 
          + stk__.state_code     + '\n' 
          + stk__.diff_code      + '\n'
          + stk__.parameter_code + '\n' 
          + stk__.advection_code + '\n'
          + stk__.diffusion_code +'\n'
          + stk__.boundary_condition_code
          )
rv    = 'A, B, C, D, _d_A, _d_D, _d_C, _d_B'

code  = template.format(body       = code,
                        return_val = rv
                        )
# print(code)

scope = {}
exec(code, {**stk__.rhs_functions}, scope)
# print(scope.keys())

model__      = njit(scope['model_M2'])
time__       = 0
states__     = stk__.expand_init(np.array([1, 2, 3, 4]))
parameters__ = np.array([0, 0, 0, 0, 0, 2, 2, 3, 1, 1, 1])

print('Calling M2')
r = model__(time__, states__, parameters__, **stk__.args)

###############################################################################
#Test RHSDct
###############################################################################
#Boundary
code  = (stk.rhsdef           + '\n' 
         + stk.state_code     + '\n' 
         + stk.diff_code      + '\n'
         + stk.parameter_code + '\n' 
         + stk.boundary_condition_code
         )
code  = template.format(body       = code,
                        return_val = 'A, B, C, D, _bc_C, _d_C'
                        )

# print(code)

scope = {}
exec(code, {**stk.rhsdct_functions}, scope)
# print(scope.keys())

model      = njit(scope['model_M0'])
time       = np.array([0, 1])
states     = np.array([0, 0, 0, 0,
                       0, 0, 0, 0,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       ])
states     = np.array([states, states]).T
parameters = np.array([0, 0, 0, 0, 0, 2, 2, 3, 0, 0, 0])
parameters = np.array([parameters, parameters*2]).T

r = model(time, states, parameters, **stk.args)

#Diffusion
code  = (stk.rhsdef           + '\n' 
         + stk.state_code     + '\n' 
         + stk.diff_code      + '\n'
         + stk.parameter_code + '\n' 
         + stk.diffusion_code
         )
code  = template.format(body       = code,
                        return_val = 'A, B, C, D, _dfn_B, _dfn_C, _dfn_D, _d_A, _d_B, _d_C, _d_D'
                        )

# print(code)

scope = {}
exec(code, {**stk.rhsdct_functions}, scope)
# print(scope.keys())

model      = njit(scope['model_M0'])
time       = np.array([0, 1])
states     = np.array([4, 4, 4, 4,
                       4, 0, 0, 0,
                       4, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 4,
                       4, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 4,
                       ])
states     = np.array([states, states]).T
parameters = np.array([0, 0, 0, 0, 0, 2, 2, 3, 0, 0, 0])
parameters = np.array([parameters, parameters*2]).T

r = model(time, states, parameters, **stk.args)

#Advection
code  = (stk.rhsdef           + '\n' 
         + stk.state_code     + '\n' 
         + stk.diff_code      + '\n'
         + stk.parameter_code + '\n' 
         + stk.advection_code
         )
code  = template.format(body       = code,
                        return_val = 'A, B, C, D, _adv_B, _adv_C, _adv_D, _d_A, _d_B, _d_C, _d_D'
                        )

# print(code)

scope = {}
exec(code, {**stk.rhsdct_functions}, scope)
# print(scope.keys())

model      = scope['model_M0']#njit(scope['model_M0'])
time       = 0
states     = np.array([4, 4, 4, 4,
                       4, 4, 4, 4,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                       ])
states     = np.array([states, states]).T
parameters = np.array([0, 0, 0, 0, 0, 0, 0, 0, -2, 1, -1])
parameters = np.array([parameters, parameters*2]).T

r = model(time, states, parameters, **stk.args)
