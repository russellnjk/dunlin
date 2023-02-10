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
z_axis = np.linspace(*span, interval)

#Grid
X, Y, Z = np.meshgrid(x_axis, y_axis, z_axis, sparse=False)
grid = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

def scatter_3D(shape, ax, grid=grid):
    s = 5
    
    is_inside = shape.contains_points(grid)
    interior  = grid[is_inside]
    
    ax.scatter(interior[:,0], interior[:,1], interior[:,2], color='orange', s=s)
    
    return 

plt.close('all')
fig = plt.figure(figsize=(18, 10))
AX  = []
for i in range(6):
    ax  = fig.add_subplot(2, 3, i+1, projection='3d')
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

###############################################################################
#Test Cube
###############################################################################
shape0 = csg.Cube()
points = np.array([[0, 0, 0],
                   [1.1, 1.1, 1.1],
                   [0.8, 0.2, 0.2],
                   [-1, -1, -1],
                   [-1, -2, -1]
                   ])

# print(shape0.contains_points(points))
assert all(shape0.contains_points(points) == [True, False, True, True, False])
scatter_3D(shape0, AX[0])

transformations = [['scale', 1, 0.5, 1],
                   ['rotate', np.pi/4, 0, 0, 1], 
                   ]

shape1 = shape0
for transformation in transformations:
    shape1 = shape1(*transformation)
scatter_3D(shape1, AX[1])

shape3 = csg.Cube()
shape4 = shape3('translate', 1.8, 1.8, 0)
shape5 = csg.Composite('union', shape3, shape4)
shape6 = csg.Composite('intersection', shape3, shape4)
shape7 = csg.Composite('difference', shape3, shape4)

scatter_3D(shape5, AX[2])
scatter_3D(shape6, AX[3])
scatter_3D(shape7, AX[4])

points  = np.array([[0, 0, 0],
                    [0.7, 0.7, 0],
                    [0.9, 0.9, 0],
                    [1.2, 1.2, 0],
                    [-1, -2, 0]
                    ])
print(shape7.contains_points(points))
assert all(shape3.contains_points(points) == [True,  True,  True,  False, False])
assert all(shape4.contains_points(points) == [False, False, True,  True,  False])
assert all(shape5.contains_points(points) == [True,  True,  True,  True,  False])
assert all(shape6.contains_points(points) == [False, False, True,  False, False])
assert all(shape7.contains_points(points) == [True,  True,  False, False, False])

shape8 = shape3('translate', 0, 0.5, 0)
shape9 = csg.Composite('union', shape3, shape4, shape8)
shape10 = shape3('translate', 0.5, 0, 0)
shape11 = csg.Composite('intersection', shape3, shape8, shape10)

scatter_3D(shape11, AX[5])

assert all(shape9.contains_points(points) == [True,  True,  True,  True, False])

###############################################################################
#Test Plotting Methods
###############################################################################
fig = plt.figure(figsize=(18, 10))
AX  = []
for i in range(6):
    ax  = fig.add_subplot(2, 3, i+1, projection='3d')
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

shapes = [shape0, shape1, shape3, shape5, shape6, shape7]

interior_args = {'s': 5, 'color': 'orange'}
exterior_args = {'s': 5, 'color': 'green' }

for ax, shape in zip(AX, shapes):
    shape.scatter_3D(ax, 
                    step=0.1,
                    interior_args=interior_args
                    )
    shape.scatter_3D(ax, 
                    step=0.5,
                    interior=False, 
                    exterior=True,
                    exterior_args=exterior_args
                    )