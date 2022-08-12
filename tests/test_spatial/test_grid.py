import matplotlib.pyplot as plt
import numpy             as np
from numbers import Number
from typing  import Sequence

import addpath
import dunlin as dn
from dunlin.spatial.geometrydefinition.grid import (BasicGrid, NestedGrid,
                                                    make_grids_from_config,
                                                    make_basic_grids,
                                                    merge_basic_grids,
                                                    ShapeStack
                                                    )
                                                    
###############################################################################
#Test 1: Grid Generation
###############################################################################
plt.ion()
plt.close('all')
span = -1, 7
fig  = plt.figure(figsize=(18, 10))
AX   = []
for i in range(6):
    ax  = fig.add_subplot(2, 3, i+1)#, projection='3d')
    # ax.set_box_aspect((1, 1, 1))
    ax.set_box_aspect()
    ax.set_xlim(*span)
    ax.set_ylim(*span)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.grid(True)
    
    AX.append(ax)
fig.tight_layout()

grid0 = BasicGrid(1, [0, 6], [0, 6])
grid0.plot(AX[0])

grid1 = BasicGrid(0.5, [1, 3], [1, 3])
grid1.plot(AX[1])

grid2 = BasicGrid(0.5, [4, 5], [4, 5])
grid2.plot(AX[2])

grid3 = NestedGrid(grid0, grid1, grid2)
grid3.plot(AX[3])

p = 2, 2
assert grid3.graph[p] == {1: (2.5, 2.0), -1: (1.5, 2.0), 2: (2.0, 2.5), -2: (2.0, 1.5)}
p = 3, 2
assert grid3.graph[p] == {1: (4.0, 2.0), -1: (2.5, 2.0), 2: (3.0, 2.5), -2: (3.0, 1.5)}
p = 2, 3
assert grid3.graph[p] == {1: (2.5, 3.0), -1: (1.5, 3.0), 2: (2.0, 4.0), -2: (2.0, 2.5)}
p = 1, 3
assert grid3.graph[p] == {1: (1.5, 3.0), -1: (0.0, 3.0), 2: (1.0, 4.0), -2: (1.0, 2.5)}
p = 0, 3
assert grid3.graph[p] == {1: (1.0, 3.0), 2: (0.0, 4.0), -2: (0.0, 2.0)}
p = 1, 2.5
assert grid3.graph[p] == {1: (1.5, 2.5), 2: (1.0, 3.0), -2: (1.0, 2.0)}

grid4 = BasicGrid(0.25, [1.5, 2.5], [1.5, 2.5])
grid4.plot(AX[4])

grid5 = NestedGrid(grid1, grid4)
grid6 = NestedGrid(grid0, grid5, grid2)
grid6.plot(AX[5])

###############################################################################
#Test 2: Grid Generation from Config
###############################################################################
fig  = plt.figure(figsize=(18, 10))
AX   = []
for i in range(6):
    ax  = fig.add_subplot(2, 3, i+1)#, projection='3d')
    # ax.set_box_aspect((1, 1, 1))
    ax.set_box_aspect()
    ax.set_xlim(*span)
    ax.set_ylim(*span)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.grid(True)
    
    AX.append(ax)
fig.tight_layout()

grid_config = {'gr0'     : {'config': [0.5, [4, 5], [4, 5]]},
                'gr1'     : {'config': [0.25, [1.5, 2.5], [1.5, 2.5]]},
                'gr2'     : {'config': [0.5, [1, 3], [1, 3]], 
                            'children': ['gr1']
                            },
                'gr_main' : {'config': [1, [0, 6], [0, 6]], 
                            'children': ['gr0', 'gr2']
                            },
                }


basic_grids = make_basic_grids(grid_config)
# print(basic_grids)

grid = merge_basic_grids(basic_grids, grid_config, 'gr0')
# print(grid)
grid.plot(AX[0])

grid = merge_basic_grids(basic_grids, grid_config, 'gr1')
# print(grid)
grid.plot(AX[1])

grid = merge_basic_grids(basic_grids, grid_config, 'gr2')
# print(grid)
grid.plot(AX[2])

grid = merge_basic_grids(basic_grids, grid_config, 'gr_main')
# print(grid)
grid.plot(AX[3])

grid.plot(AX[4], dict(c='teal'), dict(color='coral'))

#Test master function
nested_grids = make_grids_from_config(grid_config)

###############################################################################
#Test 3: 3D Grids
###############################################################################
grid_config = {'gr0'     : {'config': [0.5, [4, 5], [4, 5], [4, 5]]},
                'gr1'     : {'config': [0.25, [1.5, 2.5], [1.5, 2.5], [1.5, 2.5]]},
                'gr2'     : {'config': [0.5, [1, 3], [1, 3], [1, 3]], 
                            'children': ['gr1']
                            },
                'gr_main' : {'config': [1, [0, 6], [0, 6], [0, 6]], 
                            'children': ['gr0', 'gr2']
                            },
                }



nested_grids = make_grids_from_config(grid_config)

fig  = plt.figure(figsize=(18, 10))
AX   = []
for i in range(1):
    ax  = fig.add_subplot(1, 1, i+1, projection='3d')
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(*span)
    ax.set_ylim(*span)
    ax.set_zlim(*span)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    plt.grid(True)
    
    AX.append(ax)
fig.tight_layout()

nested_grids['gr_main'].plot(AX[0], dict(c='ocean'), dict(color='cobalt'))

###############################################################################
#Test 4: Adding Shapes
###############################################################################
grid_config = {'gr0'     : {'config': [0.5, [4, 5], [4, 5]]},
                'gr1'     : {'config': [0.25, [1.5, 2.5], [1.5, 2.5]]},
                'gr2'     : {'config': [0.5, [1, 3], [1, 3]], 
                            'children': ['gr1']
                            },
                'gr_main' : {'config': [1, [0, 6], [0, 6]], 
                            'children': ['gr0', 'gr2']
                            },
                }

nested_grids = make_grids_from_config(grid_config)

grid = nested_grids['gr_main']

class Shape:
    def __init__(self, test_func):
        self.test_func = test_func
    
    def contains_points(self, points):
        return np.array([self.test_func(*point) for point in points])
    
#Delineate system boundaries
points    = grid.points
is_system = np.zeros(len(points), dtype=bool)


shape0 = Shape(lambda x, y: (x <= 4 and y <= 4) or (x >= 4 and 3 <= y <= 6))
shape1 = Shape(lambda x, y: 1.5 <= x <= 2.5 and 1.5 <= y <= 2.5)
shape2 = Shape(lambda x, y: 4 <= x <= 5 and 4 <= y <= 5)

stk = ShapeStack(grid, shape0, shape1, shape2)

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

scatter_args = {'c': {0: 'cobalt',
                      1: 'coral',
                      2: 'ocean',
                      3: 'purple'
                      }
                }

stk.plot_shape(AX[0], 0, **scatter_args)

stk.plot_shape(AX[1], [0, 1], **scatter_args)

stk.plot_shape(AX[2], [0, 1, 2], **scatter_args)

stk.plot_shape(AX[3], [0, 1, 2, 3], **scatter_args)

stk.plot_lines(AX[4], [0, 1], color='dark yellow')

stk.plot_lines(AX[4], [0, 1], color='dark yellow')
stk.plot_lines(AX[4], [[1, 2], [2, 3], [1, 3]], color='black')

def color(edge, name):
    if edge[0] == 0:
        return 'dark yellow'
    elif edge[0] == edge[1]:
        return 'orange'
    else:
        return 'black'

stk.plot_shape(AX[5], None, **scatter_args)
stk.plot_lines(AX[5], None, color=color)

