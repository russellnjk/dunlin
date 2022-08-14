import matplotlib.pyplot as plt
import numpy             as np

import addpath
import dunlin as dn
from dunlin.spatial.geometrydefinition.grid  import (make_grids_from_config)
from dunlin.spatial.geometrydefinition.stack import (ShapeStack,
                                                     )


###############################################################################
#Test 1: Adding Shapes
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
