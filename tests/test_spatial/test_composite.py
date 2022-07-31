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

small_interval    = 41
small_span        = 0, 2
x_axis = np.linspace(*small_span, small_interval)
y_axis = np.linspace(*small_span, small_interval)
z_axis = np.linspace(*small_span, small_interval)

#Grid
X, Y, Z = np.meshgrid(x_axis, y_axis, z_axis, sparse=False)
small_grid = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

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

#Construct primitives
fig = plt.figure(figsize=(18, 10))
AX  = []
for i in range(3):
    ax  = fig.add_subplot(1, 3, i+1, projection='3d')
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

interior_args = {'s': 5, 'color': 'orange'}

shape0 = csg.Cube()
shape1 = shape0('translate',  1.8, 1.8, 0)
shape2 = shape0('translate', -1.8, 1.8, 0)

scatter_3D(shape0, AX[0])
scatter_3D(shape1, AX[1])
scatter_3D(shape2, AX[2])

###############################################################################
#Test Union
###############################################################################
fig = plt.figure(figsize=(18, 10))
AX  = []
for i in range(4):
    ax  = fig.add_subplot(2, 2, i+1, projection='3d')
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

points = [[0, 0, 0],
          [1.8, 1.8, 0],
          [1.8, 1.8, 1.5],
          [0.9, 0.9, 0]
          ]

shape3 = csg.Composite('union', shape0, shape1, shape2)
scatter_3D(shape3, AX[0])
assert all(shape3.contains_points(points) == [True, True, False, True])

shape4 = shape3('translate', 0, 0, 1.5)
scatter_3D(shape4, AX[1])
AX[1].set_title('translate')
assert all(shape4.contains_points(points) == [False, False,True, False])

shape5 = shape3('rotate', np.pi/2, 0, 1, 0)
scatter_3D(shape5, AX[2])
AX[2].set_title('rotate')
assert all(shape5.contains_points(points) == [True, False, False, True])

shape6 = shape3('scale', 0.5, 0.5, 0.5)
scatter_3D(shape6, AX[3])
AX[3].set_title('scale')
assert all(shape6.contains_points(points) == [False, False, False, False])

###############################################################################
#Test Intersection
###############################################################################
fig = plt.figure(figsize=(18, 10))
AX  = []
for i in range(4):
    ax  = fig.add_subplot(2, 2, i+1, projection='3d')
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

shape7 = csg.Composite('intersection', shape0, shape1)
scatter_3D(shape7, AX[0])
assert all(shape7.contains_points(points) == [False, False, False, True])

shape8 = shape7('translate', 0, 0, 1.5)
scatter_3D(shape8, AX[1])
AX[1].set_title('translate')
assert all(shape8.contains_points(points) == [False, False, False, False])

shape9 = shape7('rotate', np.pi/2, 0, 1, 0)
scatter_3D(shape9, AX[2])
AX[2].set_title('rotate')
assert all(shape9.contains_points(points) == [False, False, False, False])

shape10 = shape7('scale', 0.5, 0.5, 0.5)
scatter_3D(shape10, AX[3], grid=small_grid)
AX[3].set_title('scale')
assert all(shape10.contains_points(points) == [False, False, False, True])

###############################################################################
#Test Difference
###############################################################################
fig = plt.figure(figsize=(18, 10))
AX  = []
for i in range(4):
    ax  = fig.add_subplot(2, 2, i+1, projection='3d')
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

shape11 = csg.Composite('difference', shape0, shape1)
scatter_3D(shape11, AX[0])
assert all(shape11.contains_points(points) == [True, False, False, False])

shape12 = shape11('translate', 0, 0, 1.5)
scatter_3D(shape12, AX[1])
AX[1].set_title('translate')
assert all(shape12.contains_points(points) == [False, False, False, False])

shape13 = shape11('rotate', np.pi/2, 0, 1, 0)
scatter_3D(shape13, AX[2])
AX[2].set_title('rotate')
assert all(shape13.contains_points(points) == [True, False, False, True])

shape14 = shape11('scale', 0.5, 0.5, 0.5)
scatter_3D(shape14, AX[3], grid=small_grid)
AX[3].set_title('scale')
assert all(shape14.contains_points(points) == [True, False, False, False])

