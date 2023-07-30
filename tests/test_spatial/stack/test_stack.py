import matplotlib.pyplot as plt
import numpy             as np

import addpath
import dunlin as dn
from dunlin.spatial.stack.stack import Stack

plt.close('all')
plt.ion()

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
    def __init__(self, test_func, name, domain_type):
        self.test_func   = test_func
        self.name        = name
        self.domain_type = domain_type
        
    def contains_points(self, points):
        return np.array([self.test_func(*point) for point in points])

###############################################################################
#Test Preprocessing
###############################################################################
#Set up
span = -1, 4
fig  = plt.figure(figsize=(10, 10))
AX   = []
for i in range(4):
    ax  = fig.add_subplot(2, 2, i+1)#, projection='3d')
    ax.set_box_aspect(1)
    ax.set_box_aspect()
    ax.set_xlim(*span)
    ax.set_ylim(*span)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.grid(True)
    
    AX.append(ax)

#Test class method for grid generation
grid_config  = {'step'     : 1, 
                'min'      : [0, 0], 
                'max'      : [3, 3],
                }

nested_grids = Stack.make_grids_from_config(grid_config)
nested_grids['_main'].plot(AX[0])

shape0 = Shape(lambda x, y: True,  'a', 'x')
shape1 = Shape(lambda x, y: x > 1, 'b', 'y')
shapes = [shape0, shape1]

#Test here
grid     = nested_grids['_main']
mappings = Stack._make_mappings(grid.voxels, shapes)

shape_dict            = mappings[0]
shape2domain_type     = mappings[1]
voxel2domain_type     = mappings[2]
voxel2domain_type_idx = mappings[3]
voxel2shape           = mappings[4]

assert shape2domain_type     == {'b' : 'y', 
                                 'a' : 'x'
                                 }

assert voxel2domain_type     == {(1.5, 0.5): 'y',
                                 (2.5, 0.5): 'y',
                                 (1.5, 1.5): 'y',
                                 (2.5, 1.5): 'y',
                                 (1.5, 2.5): 'y',
                                 (2.5, 2.5): 'y',
                                 (0.5, 0.5): 'x',
                                 (0.5, 1.5): 'x',
                                 (0.5, 2.5): 'x'
                                 }

assert voxel2domain_type_idx == {(1.5, 0.5): (0, 'y'),
                                 (2.5, 0.5): (1, 'y'),
                                 (1.5, 1.5): (2, 'y'),
                                 (2.5, 1.5): (3, 'y'),
                                 (1.5, 2.5): (4, 'y'),
                                 (2.5, 2.5): (5, 'y'),
                                 (0.5, 0.5): (0, 'x'),
                                 (0.5, 1.5): (1, 'x'),
                                 (0.5, 2.5): (2, 'x')
                                 }

assert voxel2shape          == {(1.5, 0.5): 'b',
                                (2.5, 0.5): 'b',
                                (1.5, 1.5): 'b',
                                (2.5, 1.5): 'b',
                                (1.5, 2.5): 'b',
                                (2.5, 2.5): 'b',
                                (0.5, 0.5): 'a',
                                (0.5, 1.5): 'a',
                                (0.5, 2.5): 'a'
                                }

assert shape_dict           == {'a': shape0, 'b': shape1}

###############################################################################
#Test Instantiation
###############################################################################
shape0 = Shape(lambda x, y:   True,                 'a', domain_type='x')
shape1 = Shape(lambda x, y:   x > 1   and  y > 1,   'b', domain_type='y')
shape2 = Shape(lambda x, y:   x < 1.5 and  y < 1.5, 'c', domain_type='x')
shape3 = Shape(lambda x, y:   x < 1   and  y > 2,   'd', domain_type='z')
shape4 = Shape(lambda x, y:   x > 2   and  y < 1 ,  'e', domain_type='z')
shapes = [shape0, shape1, shape2, shape3, shape4]

grid_config  = {'step'     : 1, 
                'min'      : [0, 0], 
                'max'      : [3, 3],
                'children' : {'child': {'step' : 0.5,
                                        'min'  : [1, 1],
                                        'max'  : [2, 2]
                                        }
                              }
                }

nested_grids = Stack.make_grids_from_config(grid_config)
nested_grids['_main'].plot(AX[1])
nested_grids['child'].plot(AX[2])

grid = nested_grids['_main']
stk  = Stack(grid, shapes)

domain_type_args = {'facecolor': {'x': 'steel',
                                  'y': 'pinkish',
                                  'z': 'grey green'
                                  }
                    }

stk.plot_voxels(AX[3], domain_type_args=domain_type_args)

###############################################################################
#Test Faulty Instantiation
###############################################################################
shape0 = Shape(lambda x, y:   True,                 'a', domain_type='x')
shape1 = Shape(lambda x, y:   x > 1   and  y > 1,   'b', domain_type='y')
shape2 = Shape(lambda x, y:   x < 1.5 and  y < 1.5, 'c', domain_type='x')
shape3 = Shape(lambda x, y:   x < 1   and  y > 2,   'd', domain_type='z')
shape4 = Shape(lambda x, y:   x > 2   and  y < -1 , 'e', domain_type='z')
shapes = [shape0, shape1, shape2, shape3, shape4]

grid = nested_grids['_main']
try:
    stk  = Stack(grid, shapes)
except ValueError:
    assert True
else:
    assert False
