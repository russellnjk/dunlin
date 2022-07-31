import matplotlib.pyplot as plt
import numpy             as np

import addpath
import dunlin as dn
import dunlin.spatial.geometrydefinition.csg as csg

plt.ion()

interval = 31
span     = -3, 3
x_axis = np.linspace(*span, interval)
y_axis = np.linspace(*span, interval)

#Grid
X, Y = np.meshgrid(x_axis, y_axis, sparse=False)
grid = np.stack([X.flatten(), Y.flatten()], axis=1)

def scatter_2D(shape, ax, grid=grid):
    s = 5
    
    is_inside = shape.contains_points(grid)
    interior  = grid[is_inside]
    exterior  = grid[~is_inside]
    
    ax.scatter(interior[:,0], interior[:,1], color='orange', s=s)
    ax.scatter(exterior[:,0], exterior[:,1], color='green', s=s)
    
    return 

plt.close('all')
fig = plt.figure(figsize=(18, 10))
AX  = []
for i in range(8):
    ax  = fig.add_subplot(2, 4, i+1)
    ax.set_box_aspect(1)
    ax.set_xlim(*span)
    ax.set_ylim(*span)

    plt.grid(True)
    
    AX.append(ax)
fig.tight_layout()

###############################################################################
#Test Square
###############################################################################
shape0 = csg.Square()
points = np.array([[0, 0],
                   [1.1, 1.1],
                   [0.8, 0.2],
                   [-1, -1],
                   [-1, -2]
                   ])

# print(shape0.contains_points(points))
assert all(shape0.contains_points(points) == [True, False, True, True, False])
scatter_2D(shape0, AX[0])

transformations = [['scale', 1, 0.5],
                   ['rotate', np.pi/4, 0, 0], 
                   ]

shape1 = shape0
for transformation in transformations:
    shape1 = shape1(*transformation)
scatter_2D(shape1, AX[1])
assert np.isclose(shape1.orientation, np.pi/4)

transformations = [['scale', 1, 0.5],
                    ['rotate', np.pi/4, 0, 0], 
                    ['rotate', np.pi/4, 1, 1], 
                    ]

shape2 = shape0
for transformation in transformations:
    shape2 = shape2(*transformation)
scatter_2D(shape2, AX[2])

shape3 = csg.Square()
shape4 = shape0('translate', 1.8, 1.8)
shape5 = csg.Composite('union', shape3, shape4)
shape6 = csg.Composite('intersection', shape3, shape4)
shape7 = csg.Composite('difference', shape3, shape4)

scatter_2D(shape5, AX[3])
scatter_2D(shape6, AX[4])
scatter_2D(shape7, AX[5])

points  = np.array([[0, 0],
                    [0.7, 0.7],
                    [0.9, 0.9],
                    [1.2, 1.2],
                    [-1, -2]
                    ])
print(shape7.contains_points(points))
assert all(shape3.contains_points(points) == [True,  True,  True,  False, False])
assert all(shape4.contains_points(points) == [False, False, True,  True,  False])
assert all(shape5.contains_points(points) == [True,  True,  True,  True,  False])
assert all(shape6.contains_points(points) == [False, False, True,  False, False])
assert all(shape7.contains_points(points) == [True,  True,  False, False, False])

shape8  = shape3('translate', 0, 0.5)
shape9  = csg.Composite('union', shape3, shape4, shape8)
shape10 = shape3('translate', 0.5, 0)
shape11 = csg.Composite('intersection', shape3, shape8, shape10)

scatter_2D(shape9, AX[6])
scatter_2D(shape11, AX[7])

assert all(shape9.contains_points(points) == [True,  True,  True,  True, False])

###############################################################################
#Test Plotting Methods
###############################################################################
fig = plt.figure(figsize=(18, 10))
AX  = []
for i in range(8):
    ax  = fig.add_subplot(2, 4, i+1)
    ax.set_box_aspect(1)
    ax.set_xlim(*span)
    ax.set_ylim(*span)

    plt.grid(True)
    
    AX.append(ax)
fig.tight_layout()

shapes = [shape0, shape1, shape2, shape5, shape6, shape7, shape9, shape11]

interior_args = {'s': 5, 'color': 'orange'}
exterior_args = {'s': 5, 'color': 'green' }

for ax, shape in zip(AX, shapes):
    shape.scatter_2D(ax, 
                    exterior=True, 
                    interior_args=interior_args, 
                    exterior_args=exterior_args
                    )
