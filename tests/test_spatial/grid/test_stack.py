import matplotlib.pyplot as plt
import numpy             as np

import addpath
import dunlin as dn
from dunlin.spatial.grid.grid  import (make_grids_from_config)
from dunlin.spatial.grid.stack import (Stack,
                                       )

plt.close('all')
plt.ion()

###############################################################################
#Instantiation Algorithm Tests
###############################################################################
grid_voxels = {(0, 0)   : { 1: [(1, 0.5), (1, -0.5)]},
               (2, 0)   : {-1: [(1, 0.5), (1, -0.5)]},
               (1, 0.5) : { 1:[(2, 0)],
                           -1:[(0, 0)]
                           },
               (1, -0.5): { 1:[(2, 0)],
                           -1:[(0, 0)]
                           }
               }

class Shape:
    def __init__(self, test_func, name):
        self.test_func = test_func
        self.name      = name
        
    def contains_points(self, points):
        return np.array([self.test_func(*point) for point in points])

shape0 = Shape(lambda x, y: x == 1 and y ==  0.5, 'a')
shape1 = Shape(lambda x, y: x == 1 and y == -0.5, 'b')
shapes =  [shape0, shape1]

voxel2shape, shape2voxel, shape_dict = Stack._map_shape_2_voxel(grid_voxels, 
                                                                shapes
                                                                )

assert voxel2shape == {(1.0, -0.5): 'b', (1.0, 0.5): 'a'}
assert shape2voxel == {'b': {(1.0, -0.5)}, 'a': {(1.0, 0.5)}}
assert shape_dict  == {'a': shape0, 'b': shape1}

###############################################################################
#Test 1: Adding Shapes
###############################################################################
grid_config = {'gr_main' : {'config': [1, [0, 6], [0, 6]], 
                            'children': ['gr0', 'gr2']
                            },
               'gr0'     : {'config': [0.5, [4, 5], [4, 5]]},
               'gr1'     : {'config': [0.25, [1.5, 2.5], [1.5, 2.5]]},
               'gr2'     : {'config': [0.5, [1, 3], [1, 3]], 
                            'children': ['gr1']
                            },  
                }

nested_grids = make_grids_from_config(grid_config)

grid = nested_grids['gr_main']

class Shape:
    def __init__(self, test_func, name):
        self.test_func = test_func
        self.name      = name
        
    def contains_points(self, points):
        return np.array([self.test_func(*point) for point in points])
    
#Delineate system boundaries
shape0 = Shape(lambda x, y: (x <= 4 and y <= 4) or (x >= 4 and 3 <= y <= 6), 'a')
shape1 = Shape(lambda x, y: 1.5 <= x <= 2.5 and 1.5 <= y <= 2.5, 'b')
shape2 = Shape(lambda x, y: 4 <= x <= 5 and 4 <= y <= 5, 'c')
shapes = shape0, shape1, shape2

stk = Stack(grid, *shapes)

span = -1, 7
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
fig.tight_layout()

shape_args = {'facecolor': {'a': 'cobalt', 
                            'b': 'ocean', 
                            'c': 'marigold'
                            }
              }
stk.plot_voxels(AX[0], 
                shape_args = shape_args
                )

voxels = stk.voxels
assert (1.5, 5.5) not in voxels
assert len(voxels) + 14 == len(grid.voxels)
assert voxels[1.5, 3.5] == { 1: [(2.5, 3.5)], 
                            -1: [(0.5, 3.5)], 
                            -2: [(1.25, 2.75), (1.75, 2.75)]
                            }
assert voxels[1.25, 2.25] == {-2: [(1.25, 1.75)], 
                              -1: [(0.5, 2.5)],
                               1: [(1.625, 2.125), (1.625, 2.375)],
                               2: [(1.25, 2.75)]
                              }

'''
Expected picture:
    1. One-headed arows pointing outward from the surface of the top-level shape i.e. shape a
    2. Two-headed arrows at the interfaces of the three shapes
'''


# # ###############################################################################
# # #Test 2: Instantiation from Spatial Data
# # ###############################################################################
# # from dunlin.datastructures.spatial        import SpatialModelData

# # # all_data = dn.read_dunl_file('spatial_0.dunl')
# # all_data = all_data

# # ref = 'M0'

# # spatial_data = SpatialModelData.from_all_data(all_data, ref)

# # stk = Stack.from_spatial_data(spatial_data)


