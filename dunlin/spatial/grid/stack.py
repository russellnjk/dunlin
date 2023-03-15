import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch
from collections        import namedtuple
from typing             import Union

import dunlin.utils_plot as upp
from .grid import RegularGrid, NestedGrid, make_grids_from_config

Edge     = namedtuple('Edge', 'voxel neighbour interfacial distance')
Boundary = namedtuple('Boundary', 'voxel interfacial')

class Stack:
    #For plotting
    
    default_domain_type_args = {'edgecolor': 'None'
                                }
    default_boundary_args    = {'lw': 4,
                                'color': 'black', 
                                }
    default_surface_args     = {'lw': 4,
                                'color': 'yellow', 
                                }
    
    @staticmethod
    def make_grids_from_config(grid_config):
        return make_grids_from_config(grid_config)
    
    def __init__(self, grid: Union[RegularGrid, NestedGrid], *shapes) -> None:
        #Map shapes and voxels
        mappings = self._make_mappings(grid.voxels, shapes)
        
        self.shape2domain_type = mappings[0]
        self.domain_type2shape = mappings[1]
        self.voxel2domain_type = mappings[2]
        self.domain_type2voxel = mappings[3]
        self.voxel2shape       = mappings[4]
        self.shape2voxel       = mappings[5]
        self.shape_dict        = mappings[6]
        
        #Create voxel dict for the current object
        ndims             = grid.ndims
        shifts            = [i for i in range(-ndims, ndims+1) if i]
        domain_types      = self.domain_type2voxel.keys()
        helper            = lambda : dict.fromkeys(domain_types, set())
        shift2domain_type = {i: helper() for i in shifts}
        
        self.voxels            = {}
        
        self.edge2voxel        = {}
        self.grid              = grid
        self.shifts            = shifts
        self.shift2domain_type = {}
        
        templater           = lambda : {i: {} for i in shifts}
        self.shift2bulk     = templater()
        self.shift2surface  = templater()
        self.shift2boundary = templater()
        
        for voxel in self.voxel2shape:
            #Update self.voxels
            #This method can be overriden in the subclasses to expand functionality
            self._add_voxel(voxel)
    
    @staticmethod
    def _make_mappings(voxels: dict, shapes: tuple):
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
        shape2domain_type : dict
            Maps a shape to a domain type.
        domain_type2shape : dict
            Maps a domain_type  to a shape.
        voxel2domain_type : dict
            Maps a voxel to a domain type.
        domain_type2voxel : dict
            Maps a domain type to a voxel.
        voxel2shape : dict
            Maps voxels to a shape.
        shape2voxel : dict
            Maps each shape to a set of voxels.
        shape_dict : dict
            Maps each shape name to a shape object.

        '''
        #Check shapes and map each voxel to a shape 
        voxel_array       = np.array(list(voxels))
        voxel2shape       = {}
        shape2voxel       = {}
        domain_type2shape = {}
        shape2domain_type = {}
        domain_type2voxel = {}
        voxel2domain_type = {}
        shape_dict        = {}
        
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
            
            #Map shapes and names
            shape_dict[shape_name] = shape
            
            #Map shapes and domain types
            domain_type = shape.domain_type
            
            shape2domain_type[shape_name] = domain_type
            domain_type2shape.setdefault(domain_type, set()).add(shape_name)
            
            #Map voxels and shapes/domain types
            #Determine which voxels are inside/outside the shape
            is_inside = np.array(shape.contains_points(voxel_array))
            
            voxels_inside  = voxel_array[is_inside]
            voxels_outside = voxel_array[~is_inside]
            
            #Update the mappings
            shape2voxel[shape_name] = set()
            domain_type2voxel.setdefault(domain_type, set())
            
            for voxel in voxels_inside:
                voxel = tuple(voxel)
                
                #Map voxels and domain types
                voxel2domain_type[voxel] = domain_type
                domain_type2voxel[domain_type].add(voxel)
            
                #Map the voxel and the shape
                voxel2shape[voxel]      = shape_name
                shape2voxel[shape_name].add(voxel) 
                
            #Remove the inside_voxels from voxel_array
            #Saves time on the next iteration
            voxel_array = voxels_outside
        
        if len(voxel_array):
            msg  = 'Unused voxels were detected. '
            msg += 'This could complicate the application of boundary conditions. '
            msg += 'Ensure that your largest geometry definition is the same size as the grid.'
            
            raise ValueError(msg)
            
        return (shape2domain_type,
                domain_type2shape, 
                voxel2domain_type,
                domain_type2voxel,
                voxel2shape, 
                shape2voxel, 
                shape_dict, 
                )
        
    def _add_voxel(self, voxel) -> None:
        voxel2domain_type = self.voxel2domain_type
        
        #Edge mappings
        bulks      = self.shift2bulk
        surfaces   = self.shift2surface
        boundaries = self.shift2boundary
        
        #Grid mappings
        neighbours     = self.grid.voxels[voxel]
        sizes          = self.sizes
        
        #Set up 
        domain_type0 = voxel2domain_type[voxel]
        size0        = sizes[voxel]
        
        #Template the voxel datum
        datum = {'neighbours'  : {},
                 'boundary'    : [],
                 'surface'     : {},
                 'bulk'        : {},
                 'size'        : self.sizes[voxel],
                 'shape'       : self.voxel2shape[voxel],
                 'domain_type' : domain_type0,
                 }
        
        #Update edge mappings and voxel datum
        for shift in self.shifts:
            shift_neighbours     = neighbours.get(shift, [])
            new_shift_neighbours = []
            
            for neighbour in shift_neighbours:
                #Eliminate neighbours not part of any shape
                if not voxel2domain_type.get(neighbour):
                    continue
                
                new_shift_neighbours.append(neighbour)
                
                #Determine type of edge by checking domain type
                domain_type1 = voxel2domain_type[neighbour]
                size1        = sizes[neighbour]
                interfacial  = min(size0, size1)
                distance     = size0 + size1
                edge         = {(voxel, neighbour): {'interfacial': interfacial,
                                                     'distance'   : distance
                                                     }, 
                                }
                
                #Bulk edge
                if domain_type0 == domain_type1:
                    bulks[shift].setdefault(domain_type0, {}).update(edge)
                    datum['bulk'].setdefault(shift, []).append(neighbour)
                    
                #Surface edge
                else:
                    domain_type = domain_type0, domain_type1
                    surfaces[shift].setdefault(domain_type, {}).update(edge)
                    datum['surface'].setdefault(shift, {})[domain_type1] = neighbour
                    
                # #Boundary due to neighbour being smaller
                # if size0 > size1 and len(shift_neighbours) == 1:
                #     boundary = {'size': size1
                #                 }
                    
                #     if self.ndims == 2:
                #         idx               = abs(shift) - 1
                #         delta             = [size0/2]*(idx+1) if shift > 0 else [-size0/2]*(idx+1) 
                #         delta[idx]        = 0
                        
                #         rel_dist2boundary = size0/(size0+size1)
                #         total             = np.array(voxel) + np.array(neighbour)
                #         loc               = total*rel_dist2boundary + delta
                #         boundary['loc']   = loc
                        
                #     boundaries[shift].setdefault(domain_type0, {})[voxel] = boundary
                #     datum['boundary'].append(shift)
                    
            if new_shift_neighbours:
                datum['neighbours'][shift] = new_shift_neighbours
            else:
                #Boundary due to being located at the side of the grid
                boundary = {'size'   : size0
                            }
                
                if self.ndims == 2:
                    idx   = abs(shift) - 1
                    delta = size0/2 if shift > 0 else -size0/2 
                    new   = voxel[idx] + delta 
                    loc   = voxel[:idx] + (new,) + voxel[idx+1:]
                    
                    boundary['loc'] = loc
                    
                boundaries[shift].setdefault(domain_type0, {})[voxel] = boundary
                datum['boundary'].append(shift)
                
        self.voxels[voxel] = datum
    
    def _get_vertices(self, voxel, shift):
        pass
        
    ###########################################################################
    #Attributes from Grid
    ###########################################################################
    @property
    def sizes(self) -> dict:
        return self.grid.sizes
    
    @property
    def vertices(self) -> dict:
        return self.grid.vertices
    
    @property
    def ndims(self) -> int:
        return self.grid.ndims
    
    ###########################################################################
    #Access
    ###########################################################################
    def get_edges(self, shape0, shape1):
        shape0 = shape0 if type(shape0) == str else shape0.name
        shape1 = shape1 if type(shape1) == str else shape1.name
        
        key = shape0, shape1
        
        return self.edge2voxel[key]
    
    def get_neighbours(self, voxel_center):
        neighbours = self.voxels[voxel_center]['neighbours']
        return neighbours
    
    def get_voxel_size(self, voxel):
        size = self.sizes[voxel]
        return size
    
    def get_shape(self, voxel):
        shape_name  = self.voxel2shape[voxel]
        return shape_name
    
    def plot_voxels(self, 
                    ax, 
                    skip_grid=True, 
                    domain_type_args=None,
                    boundary_args=None,
                    surface_args=None
                    ):
        #Parse plot args
        converters   = {'facecolor': upp.get_color, 
                        'surfacecolor': upp.get_color,
                        'color'    : upp.get_color
                        }
        
        default_domain_type_args    = self.default_domain_type_args
        default_boundary_args = self.default_boundary_args
        default_surface_args     = self.default_surface_args
        
        #Plot on the axes
        shape_results    = []
        boundary_results = []
        surface_results     = []
        # voxel2shape      = self.voxel2shape
        # boundaries       = self.boundaries
        
        #Make caches for plotting arguments
        domain_type_args_cache    = {}
        boundary_args_cache = {}
        surface_args_cache     = {}
        
        surface_patches     = []
        
        for voxel, datum in self.voxels.items():
            domain_type = datum['domain_type']
            
            #Determine the arguments for the shape
            domain_type_args_ = domain_type_args_cache.get(domain_type)
            
            if not domain_type_args_:
                sub_args          = {'name': domain_type, 'voxel': voxel}
                domain_type_args_ = upp.process_kwargs(domain_type_args,
                                                       [domain_type],
                                                       default_domain_type_args,
                                                       sub_args,
                                                       converters
                                                       )
                domain_type_args_cache[domain_type] = domain_type_args_
            
            #Create the patch
            if self.ndims == 3:
                raise NotImplementedError()
            else:
                size   = self.sizes[voxel]
                s      = size/2
                anchor = [i-s for i in voxel]
                patch  = Rectangle(anchor, size, size, **domain_type_args_)
                
            #Plot the patch
            temp = ax.add_patch(patch)
            shape_results.append(temp)
            
            #Look for boundaries
            boundaries = datum['boundary']
            for shift in boundaries:
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
                    
                    boundary_args_cache[domain_type] = boundary_args_
                    
                #Create the line
                size        = self.sizes[voxel]
                delta       = np.sign(shift)*size/2
                idx         = abs(shift)-1
                point       = np.array(voxel)
                point[idx] += delta
                start = np.array(voxel)
                stop  = np.array(voxel)
                
                for i, n in enumerate(voxel):
                    if i == idx:
                        start[i] += delta
                        stop[i]  += delta
                    else:
                        start[i] -= size/2
                        stop[i]  += size/2
                
                x, y = np.stack([start, stop]).T
                line = ax.plot(x, y, **boundary_args_)
                
                boundary_results.append(line)
                
            #Look for surfaces
            surfaces     = datum['surface']
            domain_type0 = datum['domain_type']
            
            for shift, neighbours in surfaces.items():
                for domain_type1, neighbour in neighbours.items():
            
                    #Determine the arguments for the surface
                    surface       = domain_type0, domain_type1
                    surface_args_ = surface_args_cache.get(surface)
                    
                    if not surface_args_:
                        sub_args   = {'name': surface, 'voxel': voxel}
                        surface_args_ = upp.process_kwargs(surface_args,
                                                        [surface],
                                                        default_surface_args,
                                                        sub_args,
                                                        converters
                                                        )
                        
                        surface_args_cache[domain_type] = surface_args_
                    
                    #Create the line
                    size        = self.sizes[voxel]
                    delta       = np.sign(shift)*size/2
                    idx         = abs(shift)-1
                    point       = np.array(voxel)
                    point[idx] += delta
                    start = np.array(voxel)
                    stop  = np.array(voxel)
                    
                    for i, n in enumerate(voxel):
                        if i == idx:
                            start[i] += delta
                            stop[i]  += delta
                        else:
                            start[i] -= size/2
                            stop[i]  += size/2
                    
                    x, y = np.stack([start, stop]).T
                    line = ax.plot(x, y, **surface_args_)
                    
                    surface_results.append(line)
                
            
        return shape_results, boundary_results, surface_results
    