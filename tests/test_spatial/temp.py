import matplotlib.pyplot as plt
import numpy             as np

import addpath
import dunlin as dn
from dunlin.spatial.geometry.voxel  import (make_grids_from_config)
from dunlin.spatial.geometry.stack import (ShapeStack,
                                           )

plt.close('all')
plt.ion()

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
    def __init__(self, test_func, name):
        self.test_func = test_func
        self.name      = name
        
    def contains_points(self, points):
        return np.array([self.test_func(*point) for point in points])
    
#Delineate system boundaries
shape0 = Shape(lambda x, y: (x <= 4 and y <= 4) or (x >= 4 and 3 <= y <= 6), 'a')

stk = ShapeStack(grid, shape0)

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

stk.plot_voxels(AX[0])
