import numpy as np
from matplotlib.patches import Rectangle
from numbers import Number
from typing  import Iterable, Union

import dunlin.utils_plot as upp
from ..grid.grid   import RegularGrid, NestedGrid, make_grids_from_config
from .bidict       import One2One, One2Many

#Types
Domain_type = Union[str, Number]
Domain      = Union[str, Number]
Voxel       = tuple[Number]

#Stack class
class Stack:
    #Expected mappings and attributes
    grid                  : Union[RegularGrid, NestedGrid]
    ndims                 : int
    shifts                : list
    sizes                 : dict[Voxel, Number]
    voxels                : dict[Voxel, dict]
    shape_dict            : One2One[str, object]
    shape2domain_type     : One2Many[str, Domain_type]
    voxel2domain_type     : One2Many[Voxel, Domain_type]
    voxel2domain_type_idx : One2One[Voxel, tuple[int, Domain_type]]
    voxel2shape           : One2Many[Voxel, str]
    
    #For plotting
    default_domain_type_args = {'edgecolor': 'None'
                                }
    default_boundary_args    = {'lw'   : 4,
                                'color': 'black', 
                                }
    default_surface_args     = {'lw'   : 4,
                                'color': 'yellow', 
                                }
    
    @staticmethod
    def make_grids_from_config(grid_config):
        return make_grids_from_config(grid_config)
    
    def __init__(self, 
                 grid   : Union[RegularGrid, NestedGrid], 
                 shapes : Iterable 
                 ) -> None:
        #Copy from grid
        ndims  = grid.ndims
        shifts = [i for i in range(-ndims, ndims+1) if i]
        
        #Set attributes required for the rest of the preprocessing
        self.grid   = grid
        self.ndims  = ndims
        self.sizes  = grid.sizes
        self.shifts = shifts
        self.shapes = list(shapes)
        
        #Map shapes and voxels
        self._preprocess()
        
        #Iterate through each voxel
        for voxel in self.voxel2shape:
            #Update self.voxels
            #This method can be overriden in the subclasses to expand functionality
            self._add_voxel(voxel)
    
    def _preprocess(self) -> None:
        #Make mappings between shapes, voxels and domain types
        grid     = self.grid
        shapes   = self.shapes 
        mappings = self._make_mappings(grid.voxels, shapes)
        
        self.shape_dict            = mappings[0]
        self.shape2domain_type     = mappings[1]
        self.voxel2domain_type     = mappings[2]
        self.voxel2domain_type_idx = mappings[3]
        self.voxel2shape           = mappings[4]
        
        #Create voxel dict for the current object
        self.voxels = {}
        
        #Create mappings between shapes and domains
        shape2domain      = {shape: i for i, shape in enumerate(self.shape_dict)}
        self.shape2domain = One2Many('shape', 'domain', shape2domain)
        
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
        voxel_array           = np.array(list(voxels))
        voxel2shape           = One2Many('voxel', 'shape')
        shape2domain_type     = One2Many('shape', 'domain_type')
        voxel2domain_type     = One2Many('voxel', 'domain_type')
        voxel2domain_type_idx = One2One('voxel', 'domain_type_idx')
        shape_dict            = One2One('shape', 'shape_object')
        domain_type_idxs      = {}
        
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
            
            #Map voxels and shapes/domain types
            #Determine which voxels are inside/outside the shape
            is_inside = np.array(shape.contains_points(voxel_array))
            
            if not len(is_inside):
                msg = f'No voxels could be associated with Shape "{shape.name}". '
                msg = f'{msg} All voxels were already assigned to previous shapes'
                raise ValueError(msg)
            
            voxels_inside  = voxel_array[is_inside]
            voxels_outside = voxel_array[~is_inside]
            
            #Update the mappings
            if not voxels_inside.size:
                msg = f'No voxels could be associated with Shape "{shape.name}".'
                raise ValueError(msg)
            
            for voxel in voxels_inside:
                voxel           = tuple(voxel)
                domain_type_idx = domain_type_idxs.setdefault(domain_type, 0)
                
                domain_type_idxs[domain_type] += 1
                
                #Map voxels to other data
                voxel2domain_type[voxel]     = domain_type
                voxel2domain_type_idx[voxel] = domain_type_idx, domain_type
                voxel2shape[voxel]           = shape_name
                
            #Remove the inside_voxels from voxel_array
            #Saves time on the next iteration
            voxel_array = voxels_outside
        
        if len(voxel_array):
            msg  = 'Unused voxels were detected. '
            msg += 'Ensure that your largest geometry definition is the same size as the grid.'
            
            raise ValueError(msg)
        
        return (shape_dict,
                shape2domain_type,
                voxel2domain_type,
                voxel2domain_type_idx,
                voxel2shape, 
                )

    def _add_voxel(self, voxel) -> None:
        voxel2domain_type     = self.voxel2domain_type
        voxel2shape           = self.voxel2shape
        shape2domain          = self.shape2domain 
        
        #Grid mappings
        neighbours     = self.grid.voxels[voxel]
        
        #Set up 
        shape0       = voxel2shape[voxel]
        domain_type0 = voxel2domain_type[voxel]
        
        #Template the voxel datum
        datum = {'neighbours'  : {},
                 'boundary'    : [],
                 'surface'     : {},
                 'bulk'        : One2Many('neighbour', 'shift'),
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
                
                #Add neighbour
                new_shift_neighbours.append(neighbour)
                
                #Extract domain_type data
                domain_type1  = voxel2domain_type[neighbour]
                
                #Determine type of edge by checking domain type
                #Bulk edge
                if domain_type0 == domain_type1:
                    #Update self.voxels
                    datum['bulk'][neighbour] = shift
                    
                    #Update shape2domain
                    #Check the shape that neighbour belongs to
                    shape1 = voxel2shape[neighbour]
                    #If the neighbour belongs to a different shape
                    #Then shape0 and shape1 are part of the same domain
                    #Remap shape2domain
                    if shape0 != shape1:
                        shape2domain[shape1] = shape2domain[shape0]
                    
                    #Subclasses will extend the functionality of this method
                    self._add_bulk(voxel, neighbour, shift)
                    
                #Surface edge
                else:
                    #Update self.voxels
                    datum['surface'].setdefault(domain_type1, One2Many('neighbour', 'shift'))
                    datum['surface'][domain_type1][neighbour] = shift
                    
                    #Subclasses will extend the functionality of this method
                    self._add_surface(voxel, neighbour, shift)
                    
            if new_shift_neighbours:
                datum['neighbours'][shift] = new_shift_neighbours
            else:
                #Update self.voxels
                datum['boundary'].append(shift)
                
                #Update self.bulks
                self._add_boundary(voxel, shift)
                
        self.voxels[voxel] = datum
    
    def _add_bulk(self, voxel, neighbour, shift):
        pass
    
    def _add_surface(self, voxel, neighbour, shift):
        pass
    
    def _add_boundary(self, voxel, shift):
        pass
    
    ###########################################################################
    #Access
    ###########################################################################
    def plot_voxels(self, 
                    ax, 
                    domain_type_args = None,
                    boundary_args    = None,
                    surface_args     = None,
                    label_voxels     = True
                    ) -> tuple:
        
        if self.ndims != 2:
            msg = 'Implementation is only available for 2D case.'
            raise NotImplementedError(msg)
        
        #Parse plot args
        converters   = {'facecolor'   : upp.get_color, 
                        'surfacecolor': upp.get_color,
                        'color'       : upp.get_color
                        }
        
        default_domain_type_args = self.default_domain_type_args
        default_boundary_args    = self.default_boundary_args
        default_surface_args     = self.default_surface_args
        
        #Plot on the axes
        shape_results    = {}
        boundary_results = {}
        surface_results  = {}
        
        #Make caches for plotting arguments
        domain_type_args_cache = {}
        boundary_args_cache    = {}
        surface_args_cache     = {}
        
        for voxel, datum in self.voxels.items():
            domain_type0 = datum['domain_type']
            size0        = datum['size']
            
            #Determine the arguments for the shape
            domain_type_args_ = domain_type_args_cache.get(domain_type0)
            
            if not domain_type_args_:
                sub_args          = {'name': domain_type0, 'voxel': voxel}
                domain_type_args_ = upp.process_kwargs(domain_type_args,
                                                       [domain_type0],
                                                       default_domain_type_args,
                                                       sub_args,
                                                       converters
                                                       )
                domain_type_args_cache[domain_type0] = domain_type_args_
            
            #Create the patch
            if self.ndims == 3:
                raise NotImplementedError()
            else:
                s      = size0/2
                anchor = [i-s for i in voxel]
                patch  = Rectangle(anchor, size0, size0, **domain_type_args_)
                
            #Plot the patch
            temp = ax.add_patch(patch)
            shape_results[voxel] = temp
            
            #Add text
            if label_voxels:
                ax.text(*voxel, 
                        self.voxel2domain_type_idx[voxel][0], 
                        horizontalalignment='center'
                        )
            
            #Look for boundaries
            boundaries = datum['boundary']
            for shift in boundaries:
                #Determine the arguments for the boundary
                boundary_args_ = boundary_args_cache.get(shift)
                
                if not boundary_args_:
                    sub_args       = {'name': shift, 'voxel': voxel}
                    boundary_args_ = upp.process_kwargs(boundary_args,
                                                        [shift],
                                                        default_boundary_args,
                                                        sub_args,
                                                        converters
                                                        )
                    
                    boundary_args_cache[domain_type0] = boundary_args_
                    
                #Create the line
                delta       = np.sign(shift)*size0/2
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
                        start[i] -= size0/2
                        stop[i]  += size0/2
                
                x, y = np.stack([start, stop]).T
                line = ax.plot(x, y, **boundary_args_)
                
                boundary_results.setdefault(voxel, {})[shift] = line
            
            
            #Look for surfaces
            surfaces     = datum['surface']
            domain_type0 = datum['domain_type']
            for domain_type1, domain_type1_data in surfaces.items():
                for neighbour, shift in domain_type1_data.items():
                    key             = frozenset([voxel, neighbour])
                    neighbour_datum = self.voxels[neighbour] 
                    size1           = neighbour_datum['size']
                    domain_type1    = neighbour_datum['domain_type']
                    
                    #Prevent plotting the surface twice
                    #Plot when voxel is the smaller of the two
                    if size0 > size1:
                        continue
                    elif key in surface_results:
                        continue
                   
                    #Determine the arguments for the surface
                    surface       = frozenset([domain_type0, domain_type1])
                    surface_args_ = surface_args_cache.get(surface)
                    
                    if not surface_args_:
                        sub_args   = {'name': surface, 'voxel': (voxel, neighbour)}
                        surface_args_ = upp.process_kwargs(surface_args,
                                                           [surface],
                                                           default_surface_args,
                                                           sub_args,
                                                           converters
                                                           )
                        
                        surface_args_cache[surface] = surface_args_
                    
                    #Create the line
                    delta       = np.sign(shift)*size0/2
                    idx         = abs(shift)-1
                    point       = np.array(voxel)
                    point[idx] += delta
                    start       = np.array(voxel)
                    stop        = np.array(voxel)
                    
                    for i, n in enumerate(voxel):
                        if i == idx:
                            start[i] += delta
                            stop[i]  += delta
                        else:
                            start[i] -= size0/2
                            stop[i]  += size0/2
                    
                    x, y = np.stack([start, stop]).T
                    line = ax.plot(x, y, **surface_args_)

                    surface_results[key] = line

        return shape_results, boundary_results, surface_results
    