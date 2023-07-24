from collections import namedtuple
from numbers import Number
from typing  import Iterable, Mapping, Union

import dunlin.utils_plot as upp
from ..grid.grid   import RegularGrid, NestedGrid
from ..grid.bidict import One2One, Many2One
from .stack        import (Stack,
                           Domain_type, Domain, Voxel,
                           )

AdjacentDomains = namedtuple('AdjacentDomains',
                             'domain0 domain1 domain0_voxels domain1_voxels'
                             )
AdjacentShapes = namedtuple('AdjacentShapes',
                            'shape0_voxels shape1_voxels'
                            )

make_adjacent_shapes  = lambda : AdjacentShapes( [], [])

class DomainStack(Stack):
    #Expected mappings and attributes
    grid                  : Union[RegularGrid, NestedGrid]
    ndims                 : int
    shifts                : list
    sizes                 : dict[Voxel, Number]
    voxels                : dict[Voxel, dict]
    shape_dict            : One2One[str, object]
    shape2domain_type     : Many2One[str, Domain_type]
    voxel2domain_type     : Many2One[Voxel, Domain_type]
    voxel2domain_type_idx : One2One[Voxel, tuple[int, Domain_type]]
    voxel2shape           : Many2One[Voxel, str]
    shape2domain          : Many2One[str, Domain]
    
    grids              : dict[str, Union[RegularGrid, NestedGrid]]
    adjacent_shapes    : dict[tuple, AdjacentShapes]
    adjacent_domains   : dict[tuple, AdjacentDomains]
    voxel2domain       : Many2One[Voxel, Domain]
    domain2domain_type : Many2One[Domain, Domain_type]
    
    #For plotting
    default_domain_args = {'fontsize'            : 10,
                           'horizontalalignment' : 'center'
                           }
    
    def __init__(self, 
                 grid_config   : dict, 
                 shapes        : Iterable,
                 domain_types  : dict,
                 surfaces      : dict = None
                 ) -> None:
        
        #Generate the grids
        nested_grids = self.make_grids_from_config(grid_config)
        grid         = nested_grids['_main']
        self.grids   = nested_grids
        
        #Call the parent constructor
        self.voxel2surface_type     = Many2One('voxel', 'surface_type')
        self.voxel2surface_type_idx = One2One('voxel', 'surface_type_idx')
        self.voxel2shape_surface    = Many2One('voxel', 'shape_surface')
        shape2domain_num            = {shape.name: i for i, shape in enumerate(shapes)} 
        self.shape2domain_num       = Many2One('shape', 'domain_num', shape2domain_num)
        
        super().__init__(grid, shapes)
        
        #Reindex the domains based on user definition
        self.voxel2domain           = Many2One('voxel', 'domain')
        self.domain2domain_type_idx = {}
        self._reindex_domains(domain_types)
        
         
    
    def _reindex_surfaces(self, user_surface_types: dict) -> None:
        pass
    
    def _reindex_domains(self, user_domain_types: dict) -> None:
        grid                   = self.grid 
        voxel2domain_type_idx  = self.voxel2domain_type_idx
        voxel2domain_type      = self.voxel2domain_type
        voxel2shape            = self.voxel2shape
        shape2voxel            = voxel2shape.inverse
        old_shape2domain       = self.shape2domain
        new_shape2domain       = Many2One('shape', 'domain')
        voxel2domain           = self.voxel2domain
        domain2domain_type_idx = self.domain2domain_type_idx
        missing                = set(self.shape2domain.values())
        
        for new_domain_type, domain_type_data in user_domain_types.items():
            if 'domains' in domain_type_data:
                domains = domain_type_data['domains']
            else:
                pass
            for new_domain, internal_point in domains.items():
                
                if not grid.contains(internal_point):
                    msg = f'Internal point {internal_point} appears to be outside the grid.'
                    raise ValueError(msg)
                
                #Convert the internal point into a voxel and get its domain type
                voxel           = grid.voxelize(internal_point)
                old_domain_type = voxel2domain_type[voxel]
                
                #Check that the voxelized internal point is in the correct domain type
                if old_domain_type != new_domain_type:
                    a = f'The internal point {internal_point} '
                    b = f'belongs to domain type "{new_domain_type}". '
                    c = f'However, it was mapped to "{old_domain_type}".'
                    d = 'The internal point may be in a wrong or bad location. '
                    e = 'This is most likely solved by: '
                    f = '\n1. Choosing a different internal point OR'
                    g = '\n2. Changing the step size of the grid.'
                    h = a + b + c + d + e + f + g
                    raise ValueError(h)
                
                #Determine which shape the voxel is in
                shape = voxel2shape[voxel]
                
                #Find the old domain and all its shapes
                old_domain = old_shape2domain[shape]
                shapes     = old_shape2domain.inverse[old_domain]
                missing.remove(old_domain)
                
                #Check for overlapping domains
                if shape in new_shape2domain:
                    other = new_shape2domain[shape]
                    msg   = f'Domain {new_domain} overlaps with another domain {other}.'
                    raise ValueError(msg)
                
                #Reindex shape2domain
                #Update voxel2domain
                #Update domain2domain_type_idx
                domain2domain_type_idx.setdefault(new_domain, One2One('domain_idx', 'domain_type_idx'))
                for shape_ in shapes:
                    #Reindex shape2domain
                    new_shape2domain[shape_] = new_domain
                    
                    #Update voxel2domain
                    for voxel_ in shape2voxel[shape_]:
                        voxel2domain[voxel_]                   = new_domain
                        domain_idx2domain_type_idx             = domain2domain_type_idx[new_domain]
                        domain_type_idx                        = voxel2domain_type_idx[voxel_][0]
                        domain_idx                             = len(domain_idx2domain_type_idx)
                        domain_idx2domain_type_idx[domain_idx] = domain_type_idx
        
        #Ensure no old domains were not reindexed
        if missing:
            old_domain2shape = old_shape2domain.inverse
            missing_shapes   = [old_domain2shape[x] for x in missing]
            
            msg = f'The following shapes correspond to unassigned domains: {missing_shapes}.'
            raise ValueError(msg)
        
        self.shape2domain = new_shape2domain
        
    def _add_bulk(self, voxel, neighbour, shift) -> None:
        voxel2shape  = self.voxel2shape
        shape2domain = self.shape2domain
        
        #Update shape2domain
        #Check the shape that neighbour belongs to
        shape0 = voxel2shape[voxel]
        shape1 = voxel2shape[neighbour]
        #If the neighbour belongs to a different shape
        #Then shape0 and shape1 are part of the same domain
        #Remap shape2domain
        domain0 = shape2domain[shape0]
        if shape0 != shape1:
            shape2domain[shape1] = domain0
            
    def _add_surface(self, voxel, neighbour, shift) -> None:
        voxel2domain_type      = self.voxel2domain_type
        voxel2shape            = self.voxel2shape
        voxel2shape_surface    = self.voxel2shape_surface
        voxel2surface_type     = self.voxel2surface_type
        surface_type2voxel     = voxel2surface_type.inverse
        voxel2surface_type_idx = self.voxel2surface_type_idx
        
        #Update voxel2surface_type and voxel2surface_type_idx
        domain_type0 = voxel2domain_type[voxel]
        domain_type1 = voxel2domain_type[neighbour]
        surface_type = tuple(sorted([domain_type0, domain_type1]))
        
        #Add an if clause to prevent double computation
        if surface_type == (domain_type0, domain_type1):
            pair                         = voxel, neighbour
            voxel2surface_type[pair]     = surface_type
            surface_type_idx             = len(surface_type2voxel[surface_type])-1
            voxel2surface_type_idx[pair] = surface_type_idx
        
            #Update shape2surface
            #Check the shape that neighbour belongs to
            shape0 = voxel2shape[voxel]
            shape1 = voxel2shape[neighbour]
            
            voxel2shape_surface[pair] = shape0, shape1
        
                
    # def plot_voxels(self, 
    #                 ax, 
    #                 domain_args      = None,
    #                 **kwargs
    #                 ) -> tuple:
        
    #     #Parse plot args
    #     converters   = {'facecolor'   : upp.get_color, 
    #                     'surfacecolor': upp.get_color,
    #                     'color'       : upp.get_color
    #                     }
        
    #     shape_results, boundary_results, surface_results = super().plot_voxels(ax, 
    #                                                                            **kwargs
    #                                                                            )
    #     domain_results = []
        
    #     default_domain_args = self.default_domain_args
    #     for domain_type, domains in self.domain_types.items():
    #         for domain, internal_point in domains.items():
                
    #             domain_args_ = upp.process_kwargs(domain_args,
    #                                                 [domain_type],
    #                                                 default_domain_args,
    #                                                 {},
    #                                                 converters
    #                                                 )
                
    #             temp =  ax.text(*internal_point, domain, **domain_args_)   
    #             domain_results.append(temp)
        
        
    #     return (shape_results, boundary_results, surface_results, domain_results)
            