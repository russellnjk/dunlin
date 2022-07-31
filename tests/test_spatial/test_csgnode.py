import matplotlib.pyplot as plt
import numpy             as np

import addpath
import dunlin as dn
import dunlin.spatial.geometrydefinition.csgnode as csgn

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

interior_args = {'s': 5, 'color': 'orange'}

#Sphere only
node0 = ['sphere', ['scale', 1, 0.5, 0.5]]
node1 = node0 + [['translate', 1, 1, 0]]
node2 = node1 + [['rotate', np.pi/2, 1, 0, 0]]

shape0 = csgn.parse_node(node0)
shape0.scatter_3D(AX[0], step=0.1, interior_args=interior_args)

shape1 = csgn.parse_node(node1)
shape1.scatter_3D(AX[1], step=0.1, interior_args=interior_args)

shape2 = csgn.parse_node(node2)
shape2.scatter_3D(AX[2], step=0.1, interior_args=interior_args)

#Combine sphere and cube
node3  = ['cube', ['translate', 0, 0, 1]]
node4  = ['intersection', node3, node2]

shape3 = csgn.parse_node(node3)
shape0.scatter_3D(AX[3], step=0.1, interior_args=interior_args)

shape4 = csgn.parse_node(node4)
shape4.scatter_3D(AX[4], step=0.1, interior_args=interior_args)

node5 = ['union', node4, ['cube', ['translate', -0.75, 0, 1]]]

shape5 = csgn.parse_node(node5)
shape5.scatter_3D(AX[5], step=0.1, interior_args=interior_args)

