import numpy as np
from matplotlib.patches import Rectangle
from typing import Union

import dunlin.utils_plot as upp
from .voxel    import RegularGrid, NestedGrid, make_grids_from_config
from .csgnode import parse_node 

class ShapeStack:
    #For plotting
    _voxels = {'lw': 2, 'edgecolor': 'grey'}
    
    @classmethod
    def from_spatial_data(cls, spatial_data):
        geometry_data = spatial_data['geometry']
        
        gdefs   = geometry_data['geometry_definitions']
        shapes  = {}
        
        for gdef_name, gdef in gdefs.items():
            if gdef.definition == 'csg':
                node  = gdef.node
                dmnt  = gdef.domain_type
                shape = parse_node(node, gdef_name, dmnt)
                order = gdef.order
                
                shapes[order] = shape
            else:
                raise NotImplementedError(f'No implementation for {gdef.definition} yet.')

        #Sort the shapes
        shapes = [shapes[i] for i in sorted(shapes)]

        #Make the grid
        grid_config  = geometry_data['grid_config'].to_data()
        nested_grids = make_grids_from_config(grid_config)
        main_grid    = next(iter(nested_grids.values()))

        #Stack the shapes
        stk = cls(main_grid, *shapes)
        
        return stk

    def __init__(self, grid: Union[RegularGrid, NestedGrid], *shapes) -> None:
        voxel_centers = np.array(list(grid.voxels))
        shape_nums    = np.zeros(len(voxel_centers), dtype=np.int32) 
        num2name      = {}
        seen_names    = set()
        
        for i, shape in enumerate(shapes, start=1):
            is_inside             = shape.contains_points(voxel_centers)
            shape_nums[is_inside] = i
            
            name = shape.name
            if type(name) != str:
                msg = f'Shape names must be strings. Received "{name}" for shape {i}.'
                raise ValueError(msg)
                
            if name in seen_names:
                msg = f'Encountered more than one shape with name {name}.'
                raise ValueError(msg)
            elif name is not None:
                num2name[i] = name
                seen_names.add(name)
                
        shape_names   = [None if i == 0 else num2name[i] for i in shape_nums]
        voxel2shape_  = dict(zip(grid.voxels.keys(), shape_names))
        voxel2shape   = {}
        shape2voxel   = {}
        voxels        = {}
        grid_voxels   = grid.voxels
        
        #Iterate to get edges
        for voxel_center, shape_name in voxel2shape_.items():
            if shape_name is None:
                continue
            
            temp = {}
            for shift, shift_neighbours in grid_voxels[voxel_center].items():
                lst = [neighbour for neighbour in shift_neighbours 
                       if voxel2shape_[neighbour] is not None
                       ]
                
                if lst:
                    temp[shift] = lst
                
            voxels[voxel_center] = temp
            
            voxel2shape[voxel_center] = shape_name
            shape2voxel.setdefault(shape_name, set()).add(voxel_center)
            
        
        self.grid        = grid
        self.shapes      = shapes
        self.voxel2shape = voxel2shape
        self.shape2voxel = shape2voxel
        self.voxels      = voxels
        self.names       = [i.name for i in shapes]
        
    @property
    def sizes(self) -> dict:
        return self.grid.sizes
    
    @property
    def vertices(self) -> dict:
        return self.grid.vertices
    
    @property
    def ndims(self) -> int:
        return self.grid.ndims
    
    def get_voxels(self, shape_name):
        shape_name =  self.name2num[shape_name]
        return shape_name
    
    def get_edges(self, shape_name0, shape_name1):
        key = tuple(sorted([shape_name0, shape_name1]))
        
        #Convert frozenset to tuples
        edges = self.edges[key]
        edges = tuple(edges)
        return edges
    
    def get_neighbours(self, voxel_center):
        neighbours = self.voxels[voxel_center]
        return neighbours
    
    def get_voxel_size(self, voxel_center):
        size = self.sizes[voxel_center]
        return size
    
    def get_shape(self, voxel_center):
        shape_name  = self.voxel2shape[voxel_center]
        return shape_name
    
    def plot_voxels(self, ax, skip_grid=True, **patch_args):
        #Parse plot args
        converters   = {'facecolor': upp.get_color, 'edgecolor': upp.get_color}
        default      = self._voxels
        #Plot on the axes
        result    = []
        
        for shape_name, voxels in self.shape2voxel.items():
            for point in voxels:
                sub_args    = {'name':shape_name, 'voxel': point}
                patch_args_ = upp.process_kwargs(patch_args, 
                                                [shape_name, point], 
                                                default,
                                                sub_args, 
                                                converters
                                                )
                
                if self.ndims == 3:
                    pass
                else:
                    size   = self.sizes[point]
                    s      = size/2
                    anchor = [i-s for i in point]
                    patch  = Rectangle(anchor, size, size, **patch_args_)
                
                temp = ax.add_patch(patch)
                result.append(temp)
            
        return result
    