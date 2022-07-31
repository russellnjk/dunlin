import matplotlib.pyplot as plt
import numpy             as np
import seaborn           as sns

import addpath
import dunlin as dn
import dunlin.spatial.geometrydefinition.csgprimitive as prm

prm.square_interval = 11
prm.circle_interval = 24

plt.close('all')
span = -2, 2
fig  = plt.figure(figsize=(18, 10))
AX   = []
for i in range(6):
    if i > 1:
        ax  = fig.add_subplot(2, 3, i+1, projection='3d')
        ax.set_box_aspect((1, 1, 1))
        ax.set_zlim(*span)
    else:
        ax  = fig.add_subplot(2, 3, i+1)
        ax.set_box_aspect(1)
    
    
    ax.set_xlim(*span)
    ax.set_ylim(*span)

    plt.grid(True)
        
    AX.append(ax)
    
fig.tight_layout()

def scatter2D(points, ax):
    palette = sns.light_palette('blue', len(points))
    
    for color, point in zip(palette, points):
        ax.scatter([point[0]], [point[1]], color=color)

def scatter3D(points, ax):
    palette = sns.light_palette('blue', len(points)+1)
    
    for color, point in zip(palette[1:], points):
        ax.scatter([point[0]], [point[1]], [point[2]], color=color, s=50)
        
palette = sns.light_palette('blue', 10)
sns.palplot(palette)

points = prm.make_square()
scatter2D(points, AX[0])

points = prm.make_circle()
scatter2D(points, AX[1])

prm.circle_interval = 12
points = prm.make_sphere()
scatter3D(points, AX[2])

points = prm.make_cube()
scatter3D(points, AX[3])
