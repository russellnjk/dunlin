from time import time
import matplotlib.pyplot as plt
import numpy             as np

import addpath
import dunlin as dn
from dunlin.spatial.grid.domainstack import DomainStack as Stack

plt.close('all')
plt.ion()

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

domain_type_args = {'facecolor': {'x': 'steel',
                                  'y': 'pinkish',
                                  'z': 'grey green'
                                  }
                    }

class Shape:
    def __init__(self, test_func, name, domain_type):
        self.test_func   = test_func
        self.name        = name
        self.domain_type = domain_type
        
    def contains_points(self, points):
        return np.array([self.test_func(*point) for point in points])
    
    

#Set up
shape0 = Shape(lambda x, y: True,                 'a', domain_type='x')
shape1 = Shape(lambda x, y: x > 1   and  y > 1,   'b', domain_type='y')
shape2 = Shape(lambda x, y: x < 1.5 and  y < 1.5, 'c', domain_type='x')
shape3 = Shape(lambda x, y: x < 1   and  y > 2,   'd', domain_type='z')
shape4 = Shape(lambda x, y: x > 2   and  y < 1 ,  'e', domain_type='z')
shapes = [shape0, shape1, shape2, shape3, shape4]

grid_config  = {'step'     : 1, 
                'min'      : [0, 0], 
                'max'      : [3, 3],
                'children' : {'child': {'min'  : [1, 1],
                                        'max'  : [2, 2]
                                        }
                              }
                }

domain_types = {'x': {'dmn0': [0.5, 0.2]
                      },
                'y': {'dmn1': [2.5, 2]
                      },
                'z': {'dmn2': [0.5, 2.2],
                      'dmn3': [2.5, 0.2]
                      }
                }

adjacent_domains = {'adj0': ['dmn0', 'dmn1'],
                    'adj1': ['dmn0', 'dmn2'],
                    'adj2': ['dmn0', 'dmn3'],
                    'adj3': ['dmn1', 'dmn2'],
                    'adj4': ['dmn1', 'dmn3']
                    }

stk = Stack(grid_config, 
            shapes, 
            domain_types, 
            adjacent_domains
            )

assert stk.voxel2domain == {(1.5, 0.5)   : 'dmn0', 
                            (0.5, 1.5)   : 'dmn0', 
                            (0.5, 0.5)   : 'dmn0', 
                            (1.25, 1.25) : 'dmn0', 
                            (1.25, 1.75) : 'dmn1', 
                            (2.5, 2.5)   : 'dmn1', 
                            (1.75, 1.25) : 'dmn1', 
                            (1.5, 2.5)   : 'dmn1', 
                            (1.75, 1.75) : 'dmn1', 
                            (2.5, 1.5)   : 'dmn1', 
                            (0.5, 2.5)   : 'dmn2', 
                            (2.5, 0.5)   : 'dmn3'
                            }

stk.plot_voxels(AX[0], domain_type_args=domain_type_args)

###############################################################################
#Test Faulty Instantiation
###############################################################################
domain_types = {'x': {'dmn0': [-1, 0.45]
                      },
                'y': {'dmn1': [2.5, 2.5]
                      },
                'z': {'dmn2': [0.5, 2.5],
                      'dmn3': [2.5, 0.5]
                      }
                }

adjacent_domains = {'adj0': ['dmn0', 'dmn1'],
                    'adj1': ['dmn0', 'dmn2'],
                    'adj2': ['dmn0', 'dmn3'],
                    'adj3': ['dmn1', 'dmn2'],
                    'adj4': ['dmn1', 'dmn3']
                    }
try:
    stk = Stack(grid_config, 
                shapes, 
                domain_types, 
                adjacent_domains
                )
except:
    assert True

domain_types = {'x': {'dmn0': [0.45, 0.45]
                      },
                'y': {'dmn1': [0.5, 2.5]
                      },
                'z': {'dmn2': [0.5, 2.5],
                      'dmn3': [2.5, 0.5]
                      }
                }

adjacent_domains = {'adj0': ['dmn0', 'dmn1'],
                    'adj1': ['dmn0', 'dmn2'],
                    'adj2': ['dmn0', 'dmn3'],
                    'adj3': ['dmn1', 'dmn2'],
                    'adj4': ['dmn1', 'dmn3']
                    }

try:
    stk = Stack(grid_config, 
                shapes, 
                domain_types, 
                adjacent_domains
                )
except:
    assert True



###############################################################################
#Scale Up
###############################################################################
#Set up
shape0 = Shape(lambda x, y: y < 40, 'b', domain_type='y')
shape1 = Shape(lambda x, y: y > 60, 'c', domain_type='y')
shape2 = Shape(lambda x, y: x < 40, 'd', domain_type='z')
shape3 = Shape(lambda x, y: x > 60, 'e', domain_type='z')
shape4 = Shape(lambda x, y: 40 <= x <= 60 and 40 <= y <= 60, 'a', domain_type='x')
shapes = [shape0, shape1, shape2, shape3, shape4]

grid_config  = {'step'     : 20, 
                'min'      : [0, 0], 
                'max'      : [100, 100],
                'children' : {'child': {'min'  : [40, 40],
                                        'max'  : [60, 60]
                                        }
                              }
                }

domain_types = {'x': {'dmn0': [50, 50]
                      },
                'y': {'dmn1': [50, 80],
                      'dmn2': [50, 20]
                      },
                'z': {'dmn3': [20, 40],
                      'dmn4': [80, 40]
                      }
                }

adjacent_domains = {'adj0': ['dmn0', 'dmn1'],
                    'adj1': ['dmn0', 'dmn2'],
                    'adj2': ['dmn0', 'dmn3'],
                    'adj3': ['dmn0', 'dmn4'],
                    'adj4': ['dmn1', 'dmn3'],
                    'adj5': ['dmn1', 'dmn4'],
                    'adj6': ['dmn2', 'dmn3'],
                    'adj7': ['dmn2', 'dmn4'],
                    }

start = time()
stk = Stack(grid_config, 
            shapes, 
            domain_types, 
            adjacent_domains
            )
end = time()
print('Size:', len(stk.voxels), 'voxels.', '{:.2f}'.format(end-start), 'seconds.')

span = -20, 120
AX[1].set_xlim(*span)
AX[1].set_ylim(*span)
stk.plot_voxels(AX[2], domain_type_args=domain_type_args)

grid_config  = {'step'     : 0.5, 
                'min'      : [0, 0], 
                'max'      : [100, 100],
                'children' : {'grd1': {'min'      : [40, 40],
                                       'max'      : [60, 60],
                                       'children' : {'grd2': {'min': [45, 45],
                                                              'max': [55, 55]
                                                              }
                                                     }
                                       },
                              }
                }

start = time()
stk = Stack(grid_config, 
            shapes, 
            domain_types, 
            adjacent_domains
            )
end = time()
print('Size:', len(stk.voxels), 'voxels.', '{:.2f}'.format(end-start), 'seconds.')

span = -20, 120
AX[2].set_xlim(*span)
AX[2].set_ylim(*span)
# stk.plot_voxels(AX[2], domain_type_args=domain_type_args)

