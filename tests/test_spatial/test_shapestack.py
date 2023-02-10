import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import addpath
import dunlin as dn
from dunlin.datastructures.spatial import SpatialModelData

from dunlin.spatial.grid.stack         import Stack
from dunlin.spatial.shapestack         import ShapeStack
from dunlin.spatial.grid.grid          import make_grids_from_config
from dunlin.spatial.geometrydefinition import make_shapes

from spatial_data0 import all_data


plt.close('all')
plt.ion()

def plot_patch(ax, voxel, size, facecolor):
    anchor = voxel[0]-size/2, voxel[1]-size/2
    patch  = Rectangle(anchor, size, size, facecolor=facecolor)
    
    return ax.add_patch(patch)

span = -1, 11
fig  = plt.figure(figsize=(18, 10))
AX   = []
for i in range(6):
    ax  = fig.add_subplot(2, 3, i+1)
    ax.set_box_aspect()
    ax.set_xlim(*span)
    ax.set_ylim(*span)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.grid(True)
    
    AX.append(ax)
# fig.tight_layout()


spatial_data = SpatialModelData.from_all_data(all_data, 'M0')
shapes       = make_shapes(spatial_data['geometry_definitions'])
grids        = make_grids_from_config(spatial_data['grid_config'])
main_grid    = next(iter(grids.values()))

stk0 = Stack(main_grid, *shapes)

shape_args = {'facecolor': {'cell': 'coral',
                            'field': 'steel'
                            }
              }

stk0.plot_voxels(AX[0], 
                 skip_grid=True, 
                 shape_args=shape_args
                 )

AX[0].set_title('Reference figure')

###############################################################################
#
###############################################################################
temp = ShapeStack._map_shape_2_domain(spatial_data, *shapes)

shape2domain = temp[0] 
domain2shape = temp[1]

assert shape2domain == {'field': 'medium0', 'cell': 'cytosol0'}
assert domain2shape == {'medium0': 'field', 'cytosol0': 'cell'}

###############################################################################
#Find Boundary Condition Given a State, Voxel and List of Voxel Boundaries 
###############################################################################
#Assume the stack has been instantiated to the point that the boundaries dict 
#is available
voxel               = (1, 1)
voxel_boundaries    = stk0.boundaries.get(voxel, [])
state               = 'B'

x_bcs = ShapeStack._get_state_boundary_conditions(state, 
                                                  voxel_boundaries, 
                                                  spatial_data
                                                  ) 

#print(x_bcs)
assert x_bcs == {-2: {'condition'     : 0.1, 
                      'condition_type': 'Dirichlet'
                      }, 
                 -1: {'condition'     : -0.05, 
                      'condition_type': 'Neumann'
                      }
                 }

###############################################################################
#Find Advection Given a State, Voxel and List of Voxel Boundaries 
###############################################################################
shifts           = stk0.shifts
voxel            = (1, 1)
voxel_boundaries = stk0.boundaries.get(voxel, [])
voxel_edges      = stk0.voxel2edge.get(voxel, {})
state            = 'B'

adv_term = ShapeStack._get_advection_term(voxel,
                                          state, 
                                          voxel_boundaries,
                                          voxel_edges,
                                          shifts,
                                          spatial_data, 
                                          )

print(adv_term)
assert adv_term == {1: {'coeff': 'F_B_x'}, 2: {'coeff': 'F_B_y'}}

voxel            = (9, 9)
voxel_boundaries = stk0.boundaries.get(voxel, [])
voxel_edges      = stk0.voxel2edge.get(voxel, {})
state            = 'B'

adv_term = ShapeStack._get_advection_term(voxel,
                                          state, 
                                          voxel_boundaries,
                                          voxel_edges,
                                          shifts,
                                          spatial_data, 
                                          )

print(adv_term)
assert adv_term == {-2: {'coeff': 'F_B_y'}, -1: {'coeff': 'F_B_x'}}

for voxel in stk0.voxels:
    size             = stk0.sizes[voxel]
    shape_name       = stk0.voxel2shape[voxel]
    domain_type      = stk0.shapes[shape_name].domain_type
    voxel_boundaries = stk0.boundaries.get(voxel, [])
    voxel_edges      = stk0.voxel2edge.get(voxel, {})
    states           = spatial_data.compartments.domain_type2state[domain_type]
    
    state = 'B'
    
    #Find advection
    if state in states:
        adv_term = ShapeStack._get_advection_term(voxel,
                                                  state, 
                                                  voxel_boundaries, 
                                                  voxel_edges,
                                                  shifts,
                                                  spatial_data
                                                  )
        
        # plot_patch(AX[1], voxel, size, value=all(adv_term.values()) )
        facecolor = 'blue' if -2 in adv_term else 'red'
        plot_patch(AX[1], voxel, size, facecolor=facecolor)
        
        facecolor = 'blue' if -1 in adv_term else 'red'
        plot_patch(AX[2], voxel, size, facecolor=facecolor)
        
        facecolor = 'blue' if  1 in adv_term else 'red'
        plot_patch(AX[3], voxel, size, facecolor=facecolor)
        
        facecolor = 'blue' if  2 in adv_term else 'red'
        plot_patch(AX[4], voxel, size, facecolor=facecolor)
        
###############################################################################
#Find Diffusion Given a State, Voxel and List of Voxel Boundaries 
###############################################################################
shifts           = stk0.shifts
voxel            = (1, 1)
voxel_boundaries = stk0.boundaries.get(voxel, [])
voxel_edges      = stk0.voxel2edge.get(voxel, {})
state            = 'B'

dfn_term = ShapeStack._get_diffusion_term(voxel,
                                          state, 
                                          voxel_boundaries,
                                          voxel_edges,
                                          shifts,
                                          spatial_data, 
                                          )

print(dfn_term)
assert dfn_term == {1: {'coeff': 'J_B_x'}, 2: {'coeff': 'J_B_y'}}

voxel            = (9, 9)
voxel_boundaries = stk0.boundaries.get(voxel, [])
voxel_edges      = stk0.voxel2edge.get(voxel, {})
state            = 'B'

dfn_term = ShapeStack._get_advection_term(voxel,
                                          state, 
                                          voxel_boundaries,
                                          voxel_edges,
                                          shifts,
                                          spatial_data, 
                                          )

print(dfn_term)
assert dfn_term == {-2: {'coeff': 'F_B_y'}, -1: {'coeff': 'F_B_x'}}

# ###############################################################################
# #Plot to Check
# ###############################################################################


# for voxel in stk0.voxels:
#     size             = stk0.sizes[voxel]
#     shape_name       = stk0.voxel2shape[voxel]
#     domain_type      = stk0.shapes[shape_name].domain_type
#     voxel_boundaries = stk0.boundaries.get(voxel, [])
#     states           = spatial_data.compartments.domain_type2state[domain_type]
    
#     #Find boundary conditions
#     state = 'B'
#     if state in states:
#         x_bcs = ShapeStack._get_state_boundary_conditions(state, 
#                                                           voxel_boundaries, 
#                                                           spatial_data
#                                                           )
        
#         plot_patch(AX[1], voxel, size, value=x_bcs)
    
#     #Find advection
#     state = 'H'
#     if state in states:
#         x_adv = ShapeStack._get_state_advection(state, 
#                                                 voxel_boundaries, 
#                                                 voxel_edges,
#                                                 spatial_data
#                                                 )
        
#         plot_patch(AX[2], voxel, size, value=all(x_adv.values()) )
    
#     #Find diffusion
#     state = 'H'
#     if state in states:
#         x_dfn = ShapeStack._get_state_diffusion(state, 
#                                                 voxel_boundaries, 
#                                                 voxel_edges,
#                                                 spatial_data
#                                                 )
#         plot_patch(AX[3], voxel, size, value=all(x_adv.values()) )
    
        
    

# ###############################################################################
# #Find Advection For a Given State
# ###############################################################################

# # ###############################################################################
# # #
# # ###############################################################################
# # stk1 = ShapeStack(spatial_data)

# # voxel2all = stk1.voxel2all

# # assert len(voxel2all) == 100

# # bcs = spatial_data.boundary_conditions

# # e2a = stk1.element2all
# # v2a = stk1.voxel2all
# # idx = v2a[1, 1]['states']['B']

# # e = e2a[idx]
# # for k, v in e.items():
# #     print(k, ':', v)

# # assert e2a[idx]['boundaries'] == {-2: {'condition'      : 0.1, 
# #                                        'condition_type' : 'Dirichlet'
# #                                        }, 
# #                                   -1: {'condition'      : -0.05, 
# #                                        'condition_type' : 'Neumann'
# #                                        }
# #                                   }


# # assert e2a['advection'] == {1: 'F_B_x', 2: 'F_B_y'}
# # assert e2a['diffusion'] == {1: 'J_B_x', 2: 'J_B_y'}

# #

# # stk = Stack(main_grid, *shapes)


# # stk = ShapeStack(spatial_data)

# # domain_type2state = spatial_data.compartments.domain_type2state

# # voxel = ()

# # voxel = 
# '''
# For a given voxel,

# Make bulk reactions

# For neighbour
# if voxel is at boundary:
#     pass
# else:
    
# 2. 

# '''

