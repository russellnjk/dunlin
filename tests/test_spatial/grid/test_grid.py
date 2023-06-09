import matplotlib.pyplot as plt
import numpy             as np


import addpath
import dunlin as dn
from dunlin.spatial.grid.grid import (RegularGrid, 
                                      NestedGrid,
                                      make_grids_from_config
                                      )

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

#Test RegularGrid
grid0 = RegularGrid(1, [0, 6], [0, 6], name='grid0')
grid0.plot(AX[0])

voxels = grid0.voxels
assert (1.5, 1.5) in voxels
assert voxels[1.5, 1.5][-1] == [(0.5, 1.5), ]
assert voxels[1.5, 1.5][ 1] == [(2.5, 1.5), ]
assert voxels[1.5, 1.5][-2] == [(1.5, 0.5), ]
assert voxels[1.5, 1.5][ 2] == [(1.5, 2.5), ]
assert 2 not in grid0.voxels[1.5, 5.5]

assert grid0.voxelize((0.8, 0.8)) == (0.5, 0.5)
assert grid0.shift_point((1, 1), 1) == (2, 1)

assert grid0.contains((3, 3))
assert not grid0.contains((3, 9))

grid1 = RegularGrid(0.5, [1, 3], [1, 3], name='grid1')
grid1.plot(AX[1])

voxels = grid1.voxels
assert (1.25, 1.25) in voxels
assert -1 not in voxels[1.25, 1.25]
assert voxels[1.25, 1.25][ 1] == [(1.75, 1.25), ]
assert -2 not in voxels[1.25, 1.25]
assert voxels[1.25, 1.25][ 2] == [(1.25, 1.75), ]

assert grid1.voxelize((1.85, 1.85)) == (1.75, 1.75)
assert grid1.shift_point((1, 1), -1) == (0.5, 1)

#Test Nested Grid
grid2 = NestedGrid(grid0, grid1, name='grid2')
grid2.plot(AX[2])

#Test multiply nested
grid3 = RegularGrid(0.25, [1.5, 2.5], [1.5, 2.5], name='grid3')
grid3.plot(AX[3])

grid4 = NestedGrid(grid2, grid3, name='grid4')
grid4.plot(AX[4])

#Test multiple children
grid5 = RegularGrid(0.25, [4, 5], [4, 5], name='grid5')
grid5.plot(AX[3])

grid6 = NestedGrid(grid2, grid5, grid3, name='grid6')
grid6.plot(AX[5])

print(grid6.underlying)

###############################################################################
#Test Front-End Functions
###############################################################################

c = {'step': 0.5,  'min': [4,   4  ], 'max': [5,   5]} 
b = {'step': 0.25, 'min': [1.5, 1.5], 'max': [2.5, 2.5]}
a = {'step': 0.5,  'min': [1,   1  ], 'max': [3,   3  ], 'children': {'b': b}}

grid_config = {'step'     : 1,
               'min'      : [0, 0],
               'max'      : [6, 6],
               'children' : {'a': a, 'c': c}
               }

nested_grids = make_grids_from_config(grid_config)

span = -1, 7
fig  = plt.figure(figsize=(8, 8))
ax   = fig.add_subplot(1, 1, 1)
# ax.set_box_aspect((1, 1, 1))
ax.set_box_aspect()
ax.set_xlim(*span)
ax.set_ylim(*span)

ax.set_xlabel('x')
ax.set_ylabel('y')

plt.grid(True)

AX.append(ax)

nested_grids['_main'].plot(AX[6])