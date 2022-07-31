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

def scatter_3D(shape, ax, grid=grid, exterior=False):
    s = 5
    
    is_inside = shape.contains_points(grid)
    interior  = grid[is_inside]
    
    ax.scatter(interior[:,0], interior[:,1], interior[:,2], color='orange', s=s)
    
    if exterior:
        exterior = grid[~is_inside]
        ax.scatter(exterior[:,0], exterior[:,1], exterior[:,2], color='green', s=s)
        
    
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
#Test Sphere
###############################################################################
shape0 = csg.Sphere()
points = np.array([[0, 0, 0],
                   [1.1, 1.1, 0],
                   [0.4, 0.2, 0],
                   [-1, 0, 0],
                   [-1, -2, 0]
                   ])

print(shape0.contains_points(points))
assert all(shape0.contains_points(points) == [True, False, True, True, False])
scatter_3D(shape0, AX[0])

transformations = [['scale', 1, 0.5, 1],
                   ['rotate', np.pi/4, 1, 0, 0], 
                   ]

shape1 = shape0
for transformation in transformations:
    shape1 = shape1(*transformation)
scatter_3D(shape1, AX[1])

transformations = [['scale', 1, 0.5, 0.5],
                   ['rotate', np.pi/2, 0, 1, 0], 
                   ['rotate', np.pi/2, 1, 0, 0], 
                   ]

shape2 = shape0
for transformation in transformations:
    shape2 = shape2(*transformation)
scatter_3D(shape2, AX[2])#Should align with x axis

shape3 = csg.Sphere()
shape4 = shape0('translate', 0.8, 0.8, 0.8)
shape5 = csg.Composite('union', shape3, shape4)
shape6 = csg.Composite('intersection', shape3, shape4)
shape7 = csg.Composite('difference', shape3, shape4)

scatter_3D(shape5, AX[3])
scatter_3D(shape6, AX[4])
scatter_3D(shape7, AX[5])

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

shapes = [shape0, shape1, shape2, shape5, shape6, shape7]

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
