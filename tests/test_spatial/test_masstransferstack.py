import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
import numpy.ma          as ma
import textwrap          as tw
from collections import Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import addpath
import dunlin       as dn 
import dunlin.utils as ut
from dunlin.spatial.masstransferstack import (calculate_advection,
                                              calculate_diffusion,
                                              calculate_neumann_boundary,
                                              calculate_dirichlet_boundary,
                                              )
from dunlin.spatial.masstransferstack import MassTransferStack as Stack
from dunlin.datastructures.spatial    import SpatialModelData
from test_spatial_data                import all_data



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
#Test Advection Code
###############################################################################
X     = np.array([0, 10, 20, 30, 40, 50])
coeff = np.array([0, 1,   0, -1,  0, -1])

left_mapping  = np.array([[-1, 0, 1, 2, 3, -1], 
                          [ 0, 1, 1, 1, 1,  1],
                          [ 0, 1, 1, 1, 1,  0]
                          ])
right_mapping = np.array([[ 1, 2, 3, 4, -1, -1], 
                          [ 1, 1, 1, 1,  1,  0],
                          [ 1, 1, 1, 1,  0,  0]
                          ])
volumes       = np.ones(6)

dX = calculate_advection(X, coeff, (left_mapping, ), (right_mapping, ), volumes)
print(dX)

assert all(dX == [0, -10, 40, -30, 0, 0])

#Scale up
n_tiles = 200

X = np.tile(X, n_tiles)
coeff = np.array([0, 1,   0, -1,  0, -1])
coeff = np.tile(coeff, n_tiles)

h = lambda x, i: x + 6*i if x >= 0 else x
g = lambda m : [[list(map(h, m[0], [i]*6)), m[1], m[2]] for i in range(n_tiles)]
f = lambda m : np.concatenate(np.array(g(m)), axis=1)

left_mapping  = f(left_mapping)
right_mapping = f(right_mapping)
volumes       = np.ones(6*n_tiles)

dX = calculate_advection(X, coeff, (left_mapping, ), (right_mapping, ), volumes)
print(dX)

# # for key, value in stk.advection_terms.items():
# #     print(key)
# #     for k, v in value.items():
# #         print(k)
# #         print(v)
# #         print()

# # print(ut.advection_code)
code = tw.dedent(stk.advection_code)

scope = {'F_B'        : -1,
          'F_C_x'      : 1,
          'F_C_y'      : 0,
          'B'          : np.arange(4).astype(np.float64),
          'C'          : np.arange(12).astype(np.float64),
          'D'          : np.arange(12).astype(np.float64),
          ut.diff('B') : np.zeros(4).astype(np.float64),
          ut.diff('C') : np.zeros(12).astype(np.float64),
          ut.diff('D') : np.zeros(12).astype(np.float64),
          **stk.rhs_functions
          }

exec(code, None, scope)
# print(scope)

cax = make_colorbar_ax(AX[1])
stk.plot_advection(AX[1], 'B', scope[ut.adv('B')], cmap='coolwarm', colorbar_ax=cax)
AX[1].set_title(ut.adv('B'))

cax = make_colorbar_ax(AX[2])
stk.plot_advection(AX[2], 'C', scope[ut.adv('C')], cmap='coolwarm', colorbar_ax=cax)
AX[2].set_title(ut.adv('C'))

cax = make_colorbar_ax(AX[3])
stk.plot_advection(AX[3], 'D', scope[ut.adv('D')], cmap='coolwarm', colorbar_ax=cax)
AX[3].set_title(ut.adv('D'))

###############################################################################
#Test Diffusion Code
###############################################################################
X     = np.array([0, 10, 20, 30, 40, 50])
coeff = np.ones(6)

left_mapping  = np.array([[-1, 0, 1, 2, 3, -1], 
                          [ 0, 1, 1, 1, 1,  1],
                          [ 0, 1, 1, 1, 1,  0]
                          ])
volumes       = np.ones(6)

dX    = calculate_diffusion(X, coeff, (left_mapping, ), volumes)
print(dX)

assert all(dX == [10, 0, 0, 0, -10, 0])

fig, AX = make_fig(AX)

stk.plot_voxels(AX[4], domain_type_args=domain_type_args)

# print(stk.diffusion_code)
code = tw.dedent(stk.diffusion_code)

scope = {'J_B'        : 1,
          'J_C_x'      : 1,
          'J_C_y'      : 0,
          'B'          : np.arange(4).astype(np.float64),
          'C'          : np.arange(12).astype(np.float64),
          'D'          : np.arange(12).astype(np.float64),
          ut.diff('B') : np.zeros(4).astype(np.float64),
          ut.diff('C') : np.zeros(12).astype(np.float64),
          ut.diff('D') : np.zeros(12).astype(np.float64),
          **stk.rhs_functions
          }
exec(code, None, scope)
# print(scope)

cax = make_colorbar_ax(AX[5])
stk.plot_diffusion(AX[5], 'B', scope[ut.dfn('B')], cmap='coolwarm', colorbar_ax=cax)
AX[5].set_title(ut.dfn('B'))

cax = make_colorbar_ax(AX[6])
stk.plot_diffusion(AX[6], 'C', scope[ut.dfn('C')], cmap='coolwarm', colorbar_ax=cax)
AX[6].set_title(ut.dfn('C'))

cax = make_colorbar_ax(AX[7])
stk.plot_diffusion(AX[7], 'C', scope[ut.dfn('D')], cmap='coolwarm', colorbar_ax=cax)
AX[7].set_title(ut.dfn('D'))

assert Counter(scope[ut.dfn('B')]) == {3: 1, 1: 1, -1: 1, -3: 1}
assert Counter(scope[ut.dfn('C')]) == {1: 2, 0: 8, -1: 2}
assert Counter(scope[ut.dfn('D')]) == {5: 1, 0: 6, 1: 1, -2: 1, 2: 1, -1: 1, -5: 1}

###############################################################################
#Test Boundary Condition Code
###############################################################################
#Neumann
X          = np.array([0,  10, 20, 30, 40, 50])
left_flux  = np.array([5,   5,  5,  5,  5,  5])
right_flux = np.array([10, 10, 10, 10, 10, 10])

left_mapping = np.array([[0, 2],
                          [1, 1]
                          ])

right_mapping = np.array([[5],
                          [1]
                          ])
volumes       = np.ones(6)

dX  = calculate_neumann_boundary(X, left_flux,  left_mapping , volumes)
dX += calculate_neumann_boundary(X, right_flux, right_mapping, volumes)
print(dX)

assert all(dX == [5, 0, 5, 0, 0, 10])

#Dirichlet
X     = np.array([  0,  10,  20,  30,  40,  50])
coeff = np.array([  1,   1,   1,   1,   1,   1])
conc  = np.array([100, 100, 100, 100, 100, 100])

dX  = calculate_dirichlet_boundary(X, coeff, conc, left_mapping , volumes)
dX += calculate_dirichlet_boundary(X, coeff, conc, right_mapping, volumes)
print(dX)

assert all(dX == [100, 0, 80, 0, 0, 50])

#Plot
fig, AX = make_fig(AX)

stk.plot_voxels(AX[8], domain_type_args=domain_type_args)

# print(stk.boundary_condition_code)
code = tw.dedent(stk.boundary_condition_code)

scope = {'J_C_x'      : 0,
          'J_C_y'      : 0,
          'C'          : np.arange(12).astype(np.float64),
          ut.diff('C') : np.zeros(12).astype(np.float64),
          **stk.rhs_functions
          }
exec(code, None, scope)
# print(scope)

cax = make_colorbar_ax(AX[9])
stk.plot_boundary_condition(AX[9], 'C', scope[ut.bc('C')], cmap='coolwarm', colorbar_ax=cax)
AX[9].set_title(ut.bc('C') + ' Neumann only')

scope = {'J_C_x'      : 1,
          'J_C_y'      : 1,
          'C'          : np.arange(12).astype(np.float64),
          ut.diff('C') : np.zeros(12).astype(np.float64),
          **stk.rhs_functions
          }
exec(code, None, scope)
# print(scope)

cax = make_colorbar_ax(AX[10])
stk.plot_boundary_condition(AX[10], 'C', scope[ut.bc('C')], cmap='coolwarm', colorbar_ax=cax)
AX[10].set_title(ut.bc('C'))


