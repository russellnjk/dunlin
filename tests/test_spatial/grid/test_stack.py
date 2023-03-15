import matplotlib.pyplot as plt
import numpy             as np

import addpath
import dunlin as dn
from dunlin.spatial.grid.grid  import (make_grids_from_config)
from dunlin.spatial.grid.stack import Stack

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
grid_config = {'gr0' : {'config'   : [1, [-1, 1], [-1, 1]], 
                        },
               'gr1' : {'config'   : [1, [-2, 2], [-2, 2]], 
                        'children' : ['gr2']
                        },
                'gr2' : {'config': [0.5, [-1, 1], [-1, 1]], 
                         },
                }

nested_grids = Stack.make_grids_from_config(grid_config)

span = -3, 3
fig  = plt.figure(figsize=(18, 10))
AX   = []
for i in range(6):
    ax  = fig.add_subplot(2, 3, i+1)#, projection='3d')
    ax.set_box_aspect(1)
    ax.set_box_aspect()
    ax.set_xlim(*span)
    ax.set_ylim(*span)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.grid(True)
    
    AX.append(ax)

nested_grids['gr0'].plot(AX[0])
nested_grids['gr1'].plot(AX[1])
nested_grids['gr2'].plot(AX[2])

shape0 = Shape(lambda x, y: x > 0, 'a', 'x')
shape1 = Shape(lambda x, y: x < 0, 'b', 'x')
shapes = [shape0, shape1]

#Test here
grid_voxels = nested_grids['gr0'].voxels
mappings    = Stack._make_mappings(grid_voxels, shapes)

shape2domain_type = mappings[0]
domain_type2shape = mappings[1]
voxel2domain_type = mappings[2]
domain_type2voxel = mappings[3]
voxel2shape       = mappings[4]
shape2voxel       = mappings[5]
shape_dict        = mappings[6]

assert shape2domain_type == {'b' : 'x', 
                             'a' : 'x'
                             }
assert domain_type2shape == {'x' : {'b', 'a'}
                             }
assert voxel2domain_type == {(-0.5, -0.5) : 'x', 
                             (-0.5,  0.5) : 'x', 
                             ( 0.5, -0.5) : 'x', 
                             ( 0.5,  0.5) : 'x'
                             }
assert domain_type2voxel == {'x': {(-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)}
                             }

assert voxel2shape       == {(-0.5, -0.5) : 'b', 
                             (-0.5,  0.5) : 'b', 
                             ( 0.5, -0.5) : 'a', 
                             ( 0.5,  0.5) : 'a'
                             }
assert shape2voxel       == {'b': {(-0.5, -0.5), (-0.5, 0.5)}, 
                             'a': {( 0.5, -0.5), ( 0.5, 0.5)}
                             }
assert shape_dict        == {'a': shape0, 'b': shape1}

###############################################################################
#Test Instantiation
###############################################################################
shape0 = Shape(lambda x, y:   x > 0 or  y < 0, 'a', 'x')
shape1 = Shape(lambda x, y:   x < 0 and y > 0, 'c', 'y')
shape2 = Shape(lambda x, y:   x == -0.25 and  y == 0.25, 'b', 'x')
shapes = [shape0, shape1, shape2]

grid = nested_grids['gr1']
stk  = Stack(grid, *shapes)

#Check boundaries
assert not stk.shift2boundary[-2].get('y')
assert len(stk.shift2boundary[-2]['x']) == 4
assert stk.shift2boundary[-2]['x'][(-1.5, -1.5)] == {'size': 1, 'loc': (-1.5, -2.0)}
assert stk.shift2boundary[-2]['x'][(-0.5, -1.5)] == {'size': 1, 'loc': (-0.5, -2.0)}
assert stk.shift2boundary[-2]['x'][( 0.5, -1.5)] == {'size': 1, 'loc': ( 0.5, -2.0)}
assert stk.shift2boundary[-2]['x'][( 1.5, -1.5)] == {'size': 1, 'loc': ( 1.5, -2.0)} 

assert len(stk.shift2boundary[-1]['y']) == 2
assert len(stk.shift2boundary[-1]['x']) == 2
assert stk.shift2boundary[-1]['y'][(-1.5,  0.5)] == {'size': 1, 'loc': (-2.0,  0.5)}
assert stk.shift2boundary[-1]['y'][(-1.5,  1.5)] == {'size': 1, 'loc': (-2.0,  1.5)}
assert stk.shift2boundary[-1]['x'][(-1.5, -1.5)] == {'size': 1, 'loc': (-2.0, -1.5)}
assert stk.shift2boundary[-1]['x'][(-1.5, -0.5)] == {'size': 1, 'loc': (-2.0, -0.5)}

assert len(stk.shift2boundary[2]['y']) == 2
assert len(stk.shift2boundary[2]['x']) == 2
assert stk.shift2boundary[2]['y'][(-1.5, 1.5)] == {'size': 1, 'loc': (-1.5, 2.0)}
assert stk.shift2boundary[2]['y'][(-0.5, 1.5)] == {'size': 1, 'loc': (-0.5, 2.0)}
assert stk.shift2boundary[2]['x'][( 0.5, 1.5)] == {'size': 1, 'loc': (0.5, 2.0)}
assert stk.shift2boundary[2]['x'][( 1.5, 1.5)] == {'size': 1, 'loc': (1.5, 2.0)}

assert not stk.shift2boundary[1].get('y')
assert stk.shift2boundary[1]['x'][(1.5, -1.5)] == {'size': 1, 'loc': (2.0, -1.5)}
assert stk.shift2boundary[1]['x'][(1.5, -0.5)] == {'size': 1, 'loc': (2.0, -0.5)}
assert stk.shift2boundary[1]['x'][(1.5,  0.5)] == {'size': 1, 'loc': (2.0, 0.5)}
assert stk.shift2boundary[1]['x'][(1.5,  1.5)] == {'size': 1, 'loc': (2.0, 1.5)}

#Check surfaces
assert len(stk.shift2surface[-2])             == 1
assert len(stk.shift2surface[-2][('y', 'x')]) == 3
assert stk.shift2surface[-2][('y', 'x')][((-1.5,  0.5 ), (-1.5,  -0.5 ))] == {'interfacial': 1,   'distance': 2  }
assert stk.shift2surface[-2][('y', 'x')][((-0.75, 0.25), (-0.75, -0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2surface[-2][('y', 'x')][((-0.25, 0.75), (-0.25,  0.25))] == {'interfacial': 0.5, 'distance': 1.0}

assert len(stk.shift2surface[-1])             == 1
assert len(stk.shift2surface[-1][('x', 'y')]) == 3
assert stk.shift2surface[-1][('x', 'y')][((-0.25, 0.25), (-0.75, 0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2surface[-1][('x', 'y')][(( 0.5,  1.5 ), (-0.5,  1.5 ))] == {'interfacial': 1,   'distance': 2  }
assert stk.shift2surface[-1][('x', 'y')][(( 0.25, 0.75), (-0.25, 0.75))] == {'interfacial': 0.5, 'distance': 1.0}

assert len(stk.shift2surface[1])             == 1
assert len(stk.shift2surface[1][('y', 'x')]) == 3
assert stk.shift2surface[1][('y', 'x')][((-0.5,  1.5 ), (0.5,   1.5 ))] == {'interfacial': 1,   'distance': 2  }
assert stk.shift2surface[1][('y', 'x')][((-0.75, 0.25), (-0.25, 0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2surface[1][('y', 'x')][((-0.25, 0.75), (0.25,  0.75))] == {'interfacial': 0.5, 'distance': 1.0}

assert len(stk.shift2surface[2])             == 1
assert len(stk.shift2surface[2][('x', 'y')]) == 3
assert stk.shift2surface[2][('x', 'y')][((-0.25,  0.25), (-0.25, 0.75))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2surface[2][('x', 'y')][((-1.5,  -0.5 ), (-1.5,  0.5 ))] == {'interfacial': 1,   'distance': 2  }
assert stk.shift2surface[2][('x', 'y')][((-0.75, -0.25), (-0.75, 0.25))] == {'interfacial': 0.5, 'distance': 1.0}

#Check bulk
assert stk.shift2bulk[-2]['x'][((-0.25, 0.25), (-0.25, -0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[-2]['x'][((-1.5, -0.5), (-1.5, -1.5))] == {'interfacial': 1, 'distance': 2}
assert stk.shift2bulk[-2]['x'][((1.5, -0.5), (1.5, -1.5))] == {'interfacial': 1, 'distance': 2}
assert stk.shift2bulk[-2]['x'][((1.5, 0.5), (1.5, -0.5))] == {'interfacial': 1, 'distance': 2}
assert stk.shift2bulk[-2]['x'][((0.5, 1.5), (0.25, 0.75))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[-2]['x'][((0.5, 1.5), (0.75, 0.75))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[-2]['x'][((1.5, 1.5), (1.5, 0.5))] == {'interfacial': 1, 'distance': 2}
assert stk.shift2bulk[-2]['x'][((-0.75, -0.75), (-0.5, -1.5))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[-2]['x'][((-0.25, -0.75), (-0.5, -1.5))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[-2]['x'][((0.25, -0.75), (0.5, -1.5))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[-2]['x'][((0.75, -0.75), (0.5, -1.5))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[-2]['x'][((-0.75, -0.25), (-0.75, -0.75))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[-2]['x'][((-0.25, -0.25), (-0.25, -0.75))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[-2]['x'][((0.25, -0.25), (0.25, -0.75))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[-2]['x'][((0.75, -0.25), (0.75, -0.75))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[-2]['x'][((0.25, 0.25), (0.25, -0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[-2]['x'][((0.75, 0.25), (0.75, -0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[-2]['x'][((0.25, 0.75), (0.25, 0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[-2]['x'][((0.75, 0.75), (0.75, 0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[-2]['y'][((-1.5, 1.5), (-1.5, 0.5))] == {'interfacial': 1, 'distance': 2}
assert stk.shift2bulk[-2]['y'][((-0.5, 1.5), (-0.75, 0.75))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[-2]['y'][((-0.5, 1.5), (-0.25, 0.75))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[-2]['y'][((-0.75, 0.75), (-0.75, 0.25))] == {'interfacial': 0.5, 'distance': 1.0}

assert stk.shift2bulk[-1]['y'][((-0.5, 1.5), (-1.5, 1.5))] == {'interfacial': 1, 'distance': 2}
assert stk.shift2bulk[-1]['y'][((-0.75, 0.25), (-1.5, 0.5))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[-1]['y'][((-0.75, 0.75), (-1.5, 0.5))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[-1]['y'][((-0.25, 0.75), (-0.75, 0.75))] == {'interfacial': 0.5, 'distance': 1.0}

assert stk.shift2bulk[-1]['x'][((-0.5, -1.5), (-1.5, -1.5))] == {'interfacial': 1, 'distance': 2}
assert stk.shift2bulk[-1]['x'][((0.5, -1.5), (-0.5, -1.5))] == {'interfacial': 1, 'distance': 2}
assert stk.shift2bulk[-1]['x'][((1.5, -1.5), (0.5, -1.5))] == {'interfacial': 1, 'distance': 2}
assert stk.shift2bulk[-1]['x'][((1.5, -0.5), (0.75, -0.75))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[-1]['x'][((1.5, -0.5), (0.75, -0.25))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[-1]['x'][((1.5, 0.5), (0.75, 0.25))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[-1]['x'][((1.5, 0.5), (0.75, 0.75))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[-1]['x'][((1.5, 1.5), (0.5, 1.5))] == {'interfacial': 1, 'distance': 2}
assert stk.shift2bulk[-1]['x'][((-0.75, -0.75), (-1.5, -0.5))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[-1]['x'][((-0.25, -0.75), (-0.75, -0.75))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[-1]['x'][((0.25, -0.75), (-0.25, -0.75))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[-1]['x'][((0.75, -0.75), (0.25, -0.75))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[-1]['x'][((-0.75, -0.25), (-1.5, -0.5))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[-1]['x'][((-0.25, -0.25), (-0.75, -0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[-1]['x'][((0.25, -0.25), (-0.25, -0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[-1]['x'][((0.75, -0.25), (0.25, -0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[-1]['x'][((0.25, 0.25), (-0.25, 0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[-1]['x'][((0.75, 0.25), (0.25, 0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[-1]['x'][((0.75, 0.75), (0.25, 0.75))] == {'interfacial': 0.5, 'distance': 1.0}

assert stk.shift2bulk[1]['x'][((-0.25, 0.25), (0.25, 0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[1]['x'][((-1.5, -1.5), (-0.5, -1.5))] == {'interfacial': 1, 'distance': 2}
assert stk.shift2bulk[1]['x'][((-0.5, -1.5), (0.5, -1.5))] == {'interfacial': 1, 'distance': 2}
assert stk.shift2bulk[1]['x'][((0.5, -1.5), (1.5, -1.5))] == {'interfacial': 1, 'distance': 2}
assert stk.shift2bulk[1]['x'][((-1.5, -0.5), (-0.75, -0.75))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[1]['x'][((-1.5, -0.5), (-0.75, -0.25))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[1]['x'][((0.5, 1.5), (1.5, 1.5))] == {'interfacial': 1, 'distance': 2}
assert stk.shift2bulk[1]['x'][((-0.75, -0.75), (-0.25, -0.75))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[1]['x'][((-0.25, -0.75), (0.25, -0.75))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[1]['x'][((0.25, -0.75), (0.75, -0.75))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[1]['x'][((0.75, -0.75), (1.5, -0.5))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[1]['x'][((-0.75, -0.25), (-0.25, -0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[1]['x'][((-0.25, -0.25), (0.25, -0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[1]['x'][((0.25, -0.25), (0.75, -0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[1]['x'][((0.75, -0.25), (1.5, -0.5))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[1]['x'][((0.25, 0.25), (0.75, 0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[1]['x'][((0.75, 0.25), (1.5, 0.5))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[1]['x'][((0.25, 0.75), (0.75, 0.75))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[1]['x'][((0.75, 0.75), (1.5, 0.5))] == {'interfacial': 0.5, 'distance': 1.5}

assert stk.shift2bulk[1]['y'][((-1.5, 0.5), (-0.75, 0.25))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[1]['y'][((-1.5, 0.5), (-0.75, 0.75))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[1]['y'][((-1.5, 1.5), (-0.5, 1.5))] == {'interfacial': 1, 'distance': 2}
assert stk.shift2bulk[1]['y'][((-0.75, 0.75), (-0.25, 0.75))] == {'interfacial': 0.5, 'distance': 1.0}

assert stk.shift2bulk[2]['y'][((-1.5, 0.5), (-1.5, 1.5))] == {'interfacial': 1, 'distance': 2}
assert stk.shift2bulk[2]['y'][((-0.75, 0.25), (-0.75, 0.75))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[2]['y'][((-0.75, 0.75), (-0.5, 1.5))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[2]['y'][((-0.25, 0.75), (-0.5, 1.5))] == {'interfacial': 0.5, 'distance': 1.5}

assert stk.shift2bulk[2]['x'][((-1.5, -1.5), (-1.5, -0.5))] == {'interfacial': 1, 'distance': 2}
assert stk.shift2bulk[2]['x'][((-0.5, -1.5), (-0.75, -0.75))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[2]['x'][((-0.5, -1.5), (-0.25, -0.75))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[2]['x'][((0.5, -1.5), (0.25, -0.75))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[2]['x'][((0.5, -1.5), (0.75, -0.75))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[2]['x'][((1.5, -1.5), (1.5, -0.5))] == {'interfacial': 1, 'distance': 2}
assert stk.shift2bulk[2]['x'][((1.5, -0.5), (1.5, 0.5))] == {'interfacial': 1, 'distance': 2}
assert stk.shift2bulk[2]['x'][((1.5, 0.5), (1.5, 1.5))] == {'interfacial': 1, 'distance': 2}
assert stk.shift2bulk[2]['x'][((-0.75, -0.75), (-0.75, -0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[2]['x'][((-0.25, -0.75), (-0.25, -0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[2]['x'][((0.25, -0.75), (0.25, -0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[2]['x'][((0.75, -0.75), (0.75, -0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[2]['x'][((-0.25, -0.25), (-0.25, 0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[2]['x'][((0.25, -0.25), (0.25, 0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[2]['x'][((0.75, -0.25), (0.75, 0.25))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[2]['x'][((0.25, 0.25), (0.25, 0.75))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[2]['x'][((0.75, 0.25), (0.75, 0.75))] == {'interfacial': 0.5, 'distance': 1.0}
assert stk.shift2bulk[2]['x'][((0.25, 0.75), (0.5, 1.5))] == {'interfacial': 0.5, 'distance': 1.5}
assert stk.shift2bulk[2]['x'][((0.75, 0.75), (0.5, 1.5))] == {'interfacial': 0.5, 'distance': 1.5}

domain_type_args = {'facecolor': {'x': 'steel',
                                  'y': 'coral'
                                  }
                    }

stk.plot_voxels(AX[3], domain_type_args=domain_type_args)

