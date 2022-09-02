import matplotlib.pyplot as plt

import addpath
import dunlin as dn
from dunlin.datastructures.spatial import SpatialModelData
from dunlin.spatial.geometrydefinition.stack import (ShapeStack,
                                                     )
plt.close('all')
plt.ion()

all_data = dn.read_dunl_file('spatial_0.dunl')

mref = 'M0'
gref = 'Geo0'
ref  = mref, gref

spldata = SpatialModelData.from_all_data(all_data, mref, gref)
gdata   = spldata['geometry'] 

stk       = ShapeStack.from_geometry_data(gdata)
main_grid = stk.grid
shapes    = stk.shapes

span = -1, 11
fig  = plt.figure(figsize=(10, 10))
AX   = []
for i in range(4):
    ax  = fig.add_subplot(2, 2, i+1)
    ax.set_box_aspect()
    ax.set_xlim(*span)
    ax.set_ylim(*span)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.grid(True)
    
    AX.append(ax)
fig.tight_layout()


def plot_shape(shape):
    points    = main_grid.points
    is_inside = shape.contains_points(points)
    

    inside = points[is_inside].T
    AX[0].scatter(*inside, s=20)

plot_shape(shapes[0])
plot_shape(shapes[1])

colors = {(0, 1): 'None',
          (1, 2): 'black',
          (0, 2): 'None',
          (0, 0): 'None',
          (1, 1): 'cobalt',
          (2, 2): 'ocean'
          }
line_args = {'color': colors
             }

scatter_args = {'c': {0: 'gray',
                      1: 'cobalt',
                      2: 'ocean'
                      },
                's': 20
                }

stk.plot_shape(AX[1], **scatter_args)
stk.plot_edges(AX[2], **line_args)
