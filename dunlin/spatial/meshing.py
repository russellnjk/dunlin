import numpy as np

import matplotlib.pyplot as plt

from geometrydefinition import csg



interval = 61
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

shape0 = csg.Primitive.init('square')
points = np.array([[0, 0],
                   [1.1, 1.1],
                   [0.8, 0.2],
                   [-1, -1],
                   [-1, -2]
                   ])

normal = shape0._normal
offset = shape0._offset
atol = 1e-12
# print(shape0.contains_points(points))
assert all(shape0.contains_points(points) == [True, False, True, True, False])

transformations = [['scale', 1, 0.5],
                   ['rotate', np.pi/4], 
                   ]

shape1 = shape0
for transformation in transformations:
    shape1 = shape1(*transformation)

transformations = [['scale', 1, 0.5],
                   ['rotate', np.pi/4], 
                   ['rotate', np.pi/4, 1, 1], 
                   ]

shape2 = shape0
for transformation in transformations:
    shape2 = shape2(*transformation)

    
scatter_2D(shape0, AX[0])
scatter_2D(shape1, AX[1])
scatter_2D(shape2, AX[2])

shape0 = csg.Primitive.init('square')
shape1 = shape0('translate', 1.8, 1.8)
shape2 = csg.Composite('union', shape0, shape1)
shape3 = csg.Composite('intersection', shape0, shape1)
shape4 = csg.Composite('difference', shape0, shape1)

points  = np.array([[0, 0],
                    [0.7, 0.7],
                    [0.9, 0.9],
                    [1.2, 1.2],
                    [-1, -2]
                    ])
print(shape4.contains_points(points))
assert all(shape0.contains_points(points) == [True,  True,  True,  False, False])
assert all(shape1.contains_points(points) == [False, False, True,  True,  False])
assert all(shape2.contains_points(points) == [True,  True,  True,  True,  False])
assert all(shape3.contains_points(points) == [False, False, True,  False, False])
assert all(shape4.contains_points(points) == [True,  True,  False, False, False])

scatter_2D(shape2, AX[3])
scatter_2D(shape3, AX[4])
scatter_2D(shape4, AX[5])

shape5 = shape0('translate', 0, 0.5)
shape6 = csg.Composite('union', shape0, shape1, shape5)
shape7 = shape0('translate', 0.5, 0)
shape8 = csg.Composite('intersection', shape0, shape5, shape7)
assert all(shape6.contains_points(points) == [True,  True,  True,  True, False])

scatter_2D(shape6, AX[6])
scatter_2D(shape8, AX[7])

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

shapes = [shape2, shape3, shape4, shape5, shape6, shape7, shape8]

interior_args = {'s': 5, 'color': 'orange'}
exterior_args = {'s': 5, 'color': 'green' }

for ax, shape in zip(AX, shapes):
    shape.scatter2D(ax, 
                    exterior=True, 
                    interior_args=interior_args, 
                    exterior_args=exterior_args
                    )