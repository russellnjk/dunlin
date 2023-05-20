import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
import numpy.ma          as ma
import textwrap          as tw
from collections import Counter

import addpath
import dunlin       as dn 
import dunlin.utils as ut
from dunlin.spatial.masstransferstack import (calculate_advection0,
                                              calculate_advection1,
                                              calculate_diffusion,
                                              calculate_neumann_boundary0,
                                              calculate_neumann_boundary1
                                              )
from dunlin.spatial.masstransferstack import MassTransferStack as Stack
from dunlin.datastructures.spatial    import SpatialModelData
from test_spatial_data                import all_data

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

dX = calculate_advection1(X, coeff, left_mapping[None, :], right_mapping[None, :])
print(dX)

assert all(dX == [0, -10, 40, -30, 0, 0])

coeff = -1
dX    = calculate_advection0(X, coeff, left_mapping[None, :], right_mapping[None, :])
print(dX)

assert all(dX == [10, 10, 10, 10, -40, 0])

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

dX = calculate_advection1(X, coeff, left_mapping[None, :], right_mapping[None, :])
print(dX)

# #Set up
# plt.close('all')
# plt.ion()

# global_scope = {'__array': np.array, '__zeros': np.zeros}

# advection    = all_data['M0'].pop('reactions')
# spatial_data = SpatialModelData.from_all_data(all_data, 'M0')

# span = -1, 5
# fig  = plt.figure(figsize=(15, 10))
# AX   = []
# for i in range(6):
#     ax  = fig.add_subplot(2, 3, i+1)#, projection='3d')
#     ax.set_box_aspect(1)
#     ax.set_box_aspect()
#     ax.set_xlim(*span)
#     ax.set_ylim(*span)
    
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
    
#     plt.grid(True)
    
#     AX.append(ax)

# ###############################################################################
# #Test Instantiation
# ###############################################################################
# stk = Stack(spatial_data)

# domain_type_args = {'facecolor': {'cytosolic'     : 'steel',
#                                   'extracellular' : 'salmon'
#                                   }
#                     }

# stk.plot_voxels(AX[0], domain_type_args=domain_type_args)

# ###############################################################################
# #Test Advection Code
# ###############################################################################
# # for key, value in stk.advection_terms.items():
# #     print(key)
# #     for k, v in value.items():
# #         print(k)
# #         print(v)
# #         print()

# # print(ut.advion_code)
# code = tw.dedent(stk.advection_code)

# scope = {'F_B'        : -1,
#          'F_C_x'      : 1,
#          'F_C_y'      : 0,
#          'B'          : np.arange(4).astype(np.float64),
#          'C'          : np.arange(12).astype(np.float64),
#          'D'          : np.arange(12).astype(np.float64),
#          ut.diff('B') : np.zeros(4).astype(np.float64),
#          ut.diff('C') : np.zeros(12).astype(np.float64),
#          ut.diff('D') : np.zeros(12).astype(np.float64),
#          **stk.advection_calculators
#          }

# exec(code, global_scope, scope)
# # print(scope)

# cmap_and_values = {'cmap'   : 'coolwarm',
#                     'values' : [-8, 8]
#                     }

# color_func = stk.make_scaled_cmap(_default=cmap_and_values)

# state_args = {'facecolor': color_func,
#               }

# stk.plot_advection(AX[1], 'B', scope[ut.adv('B')], state_args)
# AX[1].set_title(ut.adv('B'))


# stk.plot_advection(AX[2], 'C', scope[ut.adv('C')], state_args)
# AX[2].set_title(ut.adv('C'))

# stk.plot_advection(AX[3], 'D', scope[ut.adv('D')], state_args)
# AX[3].set_title(ut.adv('D'))

# AX[4].grid(False)
# cb = mpl.colorbar.Colorbar(ax=AX[4], 
#                             cmap=color_func.cmaps['_default'], 
#                             norm=color_func.norms['_default']
#                             )   
# AX[4].set_title('Advection colormap')

# ###############################################################################
# #Test Diffusion Code
# ###############################################################################
# X     = np.array([0, 10, 20, 30, 40, 50])
# coeff = 1

# left_mapping  = np.array([[-1, 0, 1, 2, 3, -1], 
#                           [ 0, 1, 1, 1, 1,  1],
#                           [ 0, 1, 1, 1, 1,  0]
#                           ])

# dX    = calculate_diffusion(X, coeff, (left_mapping, ))
# print(dX)

# assert all(dX == [10, 0, 0, 0, -10, 0])


# fig  = plt.figure(figsize=(15, 10))
# for i in range(6):
#     ax  = fig.add_subplot(2, 3, i+1)#, projection='3d')
#     ax.set_box_aspect(1)
#     ax.set_box_aspect()
#     ax.set_xlim(*span)
#     ax.set_ylim(*span)
    
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
    
#     plt.grid(True)
    
#     AX.append(ax)

# stk.plot_voxels(AX[6], domain_type_args=domain_type_args)

# # print(stk.diffusion_code)
# code = tw.dedent(stk.diffusion_code)

# scope = {'J_B'         : 1,
#           'J_C_x'      : 1,
#           'J_C_y'      : 0,
#           'B'          : np.arange(4).astype(np.float64),
#           'C'          : np.arange(12).astype(np.float64),
#           'D'          : np.arange(12).astype(np.float64),
#           ut.diff('B') : np.zeros(4).astype(np.float64),
#           ut.diff('C') : np.zeros(12).astype(np.float64),
#           ut.diff('D') : np.zeros(12).astype(np.float64),
#           '__dfnfunc'  : calculate_diffusion
#           }
# exec(code, global_scope, scope)
# # print(scope)

# cmap_and_values = {'cmap'   : 'coolwarm',
#                    'values' : [-6, 6]
#                    }

# color_func = stk.make_scaled_cmap(_default=cmap_and_values)

# state_args = {'facecolor': color_func,
#               }

# stk.plot_diffusion(AX[7], 'B', scope[ut.dfn('B')], state_args)
# AX[7].set_title(ut.dfn('B'))

# stk.plot_diffusion(AX[8], 'C', scope[ut.dfn('C')], state_args)
# AX[8].set_title(ut.dfn('C'))

# stk.plot_diffusion(AX[9], 'C', scope[ut.dfn('D')], state_args)
# AX[9].set_title(ut.dfn('D'))

# AX[10].grid(False)
# cb = mpl.colorbar.Colorbar(ax=AX[10], 
#                             cmap=color_func.cmaps['_default'], 
#                             norm=color_func.norms['_default']
#                             )
# AX[10].set_title('Diffusion colormap')

# assert Counter(scope[ut.dfn('B')]) == {3: 1, 1: 1, -1: 1, -3: 1}
# assert Counter(scope[ut.dfn('C')]) == {1: 2, 0: 8, -1: 2}
# assert Counter(scope[ut.dfn('D')]) == {5: 1, 0: 6, 1: 1, -2: 1, 2: 1, -1: 1, -5: 1}

# ###############################################################################
# #Test Boundary Condition Code
# ###############################################################################
# X    = np.array([0, 10, 20, 30, 40, 50])
# flux = 5

# domain_type_idxs = np.array([0, 1, 5])
# scale            = np.array([1, 1, 1])
# dX = calculate_neumann_boundary0(X, flux, domain_type_idxs, scale)
# print(dX)

# assert all(dX == [5, 0, 5, 0, 0, 5])

# flux = np.array([1, 2, -3, 4, 5, 6])

# dX = calculate_neumann_boundary1(X, flux, domain_type_idxs, scale)
# print(dX)

# assert all(dX == [1, 0, -3, 0, 0, 6])



