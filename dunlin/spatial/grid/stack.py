import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch
from typing             import Union

import dunlin.utils_plot as upp
from .grid    import RegularGrid, NestedGrid

class Stack:
    #For plotting
    
    default_shape_args    = {'lw': 4, 
                             'edgecolor': 'dark grey'
                             }
    default_boundary_args = {'facecolor': 'black', 
                             'edgecolor': 'black',
                             'arrowstyle': 'simple',
                             'mutation_scale': 10
                             }
    default_edge_args     = {'facecolor': 'black', 
                             'edgecolor': 'black',
                             'arrowstyle': 'simple',
                             'mutation_scale': 10
                             }
    
    def __init__(self, grid: Union[RegularGrid, NestedGrid], *shapes) -> None:
        #Map shapes and voxels
        voxel2shape, shape2voxel, shape_dict = self._map_shape_2_voxel(grid.voxels, 
                                                                       shapes
                                                                       )
        self.shape2voxel = shape2voxel
        self.voxel2shape = voxel2shape
        self.shapes      = shape_dict 
        
        #Create voxel dict for the current object
        ndims       = grid.ndims
        shifts      = list(range(-ndims, ndims+1))
        shifts.pop(len(shifts)%2+1)
        
        self.voxels     = {}
        self.boundaries = {}
        self.edge2voxel = {}
        self.voxel2edge = {}
        self.grid       = grid
        self.shifts     = shifts
        
        for voxel in voxel2shape:
            # #Voxel is not part of any shape
            # if voxel not in voxel2shape:
            #     continue
            
            #Update self.voxels
            #This method can be overriden in the subclasses to expand functionality
            self._add_voxel(voxel)
    
    @staticmethod
    def _map_shape_2_voxel(voxels: dict, shapes: tuple):
        '''
        This method has been implemented as a static method to facilitate 
        stand-alone testing.

        Parameters
        ----------
        voxels : dict
            Use grid.voxels.
        shapes : tuple
            DESCRIPTION.

        Returns
        -------
        voxel2shape : dict
            Maps voxels to a shape.
        shape2voxel : dict
            Maps each shape to a set of voxels.
        shape_dict : dict
            Maps each shape name to a shape object.

        '''
        #Check shapes and map each voxel to a shape 
        voxel_array = np.array(list(voxels))
        voxel2shape = {}
        shape2voxel = {}
        shape_dict  = {}
        
        #Iterate backwards
        for shape in shapes[::-1]:
            #Check the shape has a valid name
            shape_name = shape.name
            if type(shape_name) != str:
                msg  = 'Shape names must be strings. '
                msg += f'Received "{shape_name}" for the following:\n {shape}.'
                raise ValueError(msg)
                
            elif shape_name in shape_dict:
                msg = f'Encountered more than one shape with name {shape_name}.'
                raise ValueError(msg)
            
            #Determine which voxels are inside/outside the shape
            is_inside = np.array(shape.contains_points(voxel_array))
            
            voxels_inside  = voxel_array[is_inside]
            voxels_outside = voxel_array[~is_inside]
            
            #Update the mappings
            shape2voxel[shape_name] = set()
            for voxel in voxels_inside:
                voxel = tuple(voxel)
                
                voxel2shape[voxel]      = shape_name
                shape2voxel[shape_name].add(voxel) 
            
            shape_dict[shape_name] = shape
            
            #Remove the inside_voxels from voxel_array
            #Saves time on the next iteration
            voxel_array = voxels_outside
          
        
        return voxel2shape, shape2voxel, shape_dict
        
    def _add_voxel(self, voxel) -> None:
        voxel2shape = self.voxel2shape
        
        #Voxel is part of a shape
        neighbours     = self.grid.voxels[voxel]
        new_neighbours = {}
        boundaries     = []
        
        for shift in self.shifts:
            shift_neighbours     = neighbours.get(shift, [])
            new_shift_neighbours = []
            
            for neighbour in shift_neighbours:
                if neighbour in voxel2shape:
                    new_shift_neighbours.append(neighbour)
                    
                    shape_name0 = voxel2shape[voxel]
                    shape_name1 = voxel2shape[neighbour]
                    
                    #Edge between two shapes
                    if shape_name0 != shape_name1:
                        key = (shape_name0, shape_name1)
                        self.edge2voxel.setdefault(key, {})[voxel] = {'neighbour' : neighbour,
                                                                      'shift'     : shift
                                                                      }
                        self.voxel2edge.setdefault(voxel, {})[shift] = {'neighbour': neighbour,
                                                                        'edge'     : key,
                                                                        }
                        
            if new_shift_neighbours:
                new_neighbours[shift] = new_shift_neighbours
            else:
                boundaries.append(shift)
                
        self.voxels[voxel] = new_neighbours
        
        if boundaries:
            self.boundaries[voxel] = boundaries
    
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
    
    def get_edges(self, shape0, shape1):
        shape0 = shape0 if type(shape0) == str else shape0.name
        shape1 = shape1 if type(shape1) == str else shape1.name
        
        key = shape0, shape1
        
        return self.edge2voxel[key]
    
    def get_neighbours(self, voxel_center):
        neighbours = self.voxels[voxel_center]
        return neighbours
    
    def get_voxel_size(self, voxel_center):
        size = self.sizes[voxel_center]
        return size
    
    def get_shape(self, voxel_center):
        shape_name  = self.voxel2shape[voxel_center]
        return shape_name
    
    def plot_voxels(self, 
                    ax, 
                    skip_grid=True, 
                    shape_args=None,
                    boundary_args=None,
                    edge_args=None
                    ):
        #Parse plot args
        converters   = {'facecolor': upp.get_color, 'edgecolor': upp.get_color}
        default_shape_args    = self.default_shape_args
        default_boundary_args = self.default_boundary_args
        default_edge_args     = self.default_edge_args
        
        #Plot on the axes
        shape_results    = []
        boundary_results = []
        edge_results     = []
        voxel2shape      = self.voxel2shape
        # boundaries       = self.boundaries
        
        #Make caches for plotting arguments
        shape_args_cache    = {}
        boundary_args_cache = {}
        edge_args_cache  = {}
        
        for voxel in self.voxels:
            shape = voxel2shape[voxel]
            
            #Determine the arguments for the shape
            shape_args_ = shape_args_cache.get(shape)
            
            if not shape_args_:
                sub_args    = {'name': shape, 'voxel': voxel}
                shape_args_ = upp.process_kwargs(shape_args,
                                                 [shape],
                                                 default_shape_args,
                                                 sub_args,
                                                 converters
                                                 )
                shape_args_cache[shape] = shape_args_
            
            #Create the patch
            if self.ndims == 3:
                raise NotImplementedError()
            else:
                size   = self.sizes[voxel]
                s      = size/2
                anchor = [i-s for i in voxel]
                patch  = Rectangle(anchor, size, size, **shape_args_)
            
            #Plot the patch
            temp = ax.add_patch(patch)
            shape_results.append(temp)
        
        #Look for boundaries
        for voxel, shifts in self.boundaries.items():
            for shift in shifts:
                #Determine the arguments for the boundary
                boundary_args_ = boundary_args_cache.get(shift)
                
                if not boundary_args_:
                    sub_args    = {'name': shift, 'voxel': voxel}
                    boundary_args_ = upp.process_kwargs(boundary_args,
                                                        [shift],
                                                        default_boundary_args,
                                                        sub_args,
                                                        converters
                                                        )
                    
                    boundary_args_cache[shape] = boundary_args_
                    
                #Create the patch
                start = voxel
                stop  = list(voxel)
                stop[abs(shift)-1] += np.sign(shift)*self.sizes[voxel] 
                patch = FancyArrowPatch(start, stop, **boundary_args_)
                
                #Plot the patch
                temp = ax.add_patch(patch)
                boundary_results.append(temp)
        
        #Look for edges
        for voxel, shifts in self.voxel2edge.items():
            for shift, dct in shifts.items():
                neighbour = dct['neighbour']
                edge      = dct['edge']
                
                #Determine the arguments for the edge
                edge_args_ = edge_args_cache.get(edge)
                
                if not edge_args_:
                    sub_args   = {'name': edge, 'voxel': voxel}
                    edge_args_ = upp.process_kwargs(edge_args,
                                                    [edge],
                                                    default_edge_args,
                                                    sub_args,
                                                    converters
                                                    )
                    
                    edge_args_cache[shape] = edge_args_
                
                #Create the patch
                start = voxel
                stop  = neighbour
                patch = FancyArrowPatch(start, stop, **edge_args_)
                
                #Plot the patch
                temp = ax.add_patch(patch)
                edge_results.append(temp)
         
        return shape_results, boundary_results, edge_results
    