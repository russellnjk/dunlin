import matplotlib.pyplot as plt

import addpath
import dunlin as dn
import dunlin.utils as ut
import dunlin.ode.ode_coder as odc
from dunlin.datastructures.spatial import SpatialModelData
from dunlin.spatial.geometrydefinition.stack import (ShapeStack,
                                                     )
plt.close('all')
plt.ion()

all_data = dn.read_dunl_file('spatial_0.dunl')

mref = 'M0'
gref = 'Geo0'
ref  = mref, gref

spatial_data  = SpatialModelData.from_all_data(all_data, mref, gref)
geometry_data = spatial_data['geometry'] 

shape_stack = ShapeStack.from_geometry_data(geometry_data)
main_grid   = shape_stack.grid

area  = {}
dist  = {}
ndims = shape_stack.ndims

for point, neighbours in shape_stack.graph.items():
    for shift in range(1, ndims+1):
    # for shift, neighbour in neighbours.items():
        # plus_neighbour
        # plus_key  = frozenset(point, neighbour)
        # minus_key =
        
        #Settle the plus neighbour first
        
        neighbour = neighbours.get(shift)
        if neighbour:
            key       = frozenset(point, neighbour)
            if ndims == 2:
                if abs(shift) == 1:
                    yspan = abs(neighbours[2][1] - neighbours[-2][1])
                    area[key] = yspan
                    
                    dist[key] = abs(point[0] - neighbour[0])
                elif abs(shift == 2):
                    xspan = 
            
            
        