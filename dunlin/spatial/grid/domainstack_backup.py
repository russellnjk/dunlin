from collections import namedtuple
from numbers import Number
from typing  import Iterable, Mapping, Union

import dunlin.utils_plot as upp
from .grid   import RegularGrid, NestedGrid
from .bidict import One2One, One2Many
from .stack  import (Stack,
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
    shape2domain_type     : One2Many[str, Domain_type]
    voxel2domain_type     : One2Many[Voxel, Domain_type]
    voxel2domain_type_idx : One2One[Voxel, tuple[int, Domain_type]]
    voxel2shape           : One2Many[Voxel, str]
    shape2domain          : One2Many[str, Domain]
    
    grids              : dict[str, Union[RegularGrid, NestedGrid]]
    adjacent_shapes    : dict[tuple, AdjacentShapes]
    adjacent_domains   : dict[tuple, AdjacentDomains]
    voxel2domain       : One2Many[Voxel, Domain]
    domain2domain_type : One2Many[Domain, Domain_type]
    
    #For plotting
    default_domain_args = {'fontsize'            : 10,
                           'horizontalalignment' : 'center'
                           }
    
    def __init__(self, 
                 grid_config      : dict, 
                 shapes           : Iterable,
                 domain_types     : Mapping,
                 adjacent_domains : Mapping
                 ) -> None:
        
        #Generate the grids
        nested_grids = self.make_grids_from_config(grid_config)
        grid         = nested_grids['_main']
        self.grids   = nested_grids
        
        #Call the parent constructor
        self.adjacent_shapes = {}
        super().__init__(grid, shapes)
        
        #Ensure there are no unexpected or unused domain_types
        domain_type2shape      = self.shape2domain_type.inverse
        expected_domain_types  = set(domain_types)
        received_domain_types  = set(domain_type2shape)
        
        self._check_domain_types(expected_domain_types, received_domain_types)
        
        #Find the user's domains and reindex
        self.domain_types       = domain_types
        self.voxel2domain       = One2Many('voxel', 'domain')
        self.domain2domain_type = One2Many('domain', 'domain_type')
        self._reindex_domains(domain_types)
        
        self.adjacent_domains = {}
        self._add_adjacent_domains(adjacent_domains)
        
    @staticmethod
    def _check_domain_types(expected_domain_types: set[Union[str, Number]], 
                            received_domain_types: set[Union[str, Number]] 
                            ) -> None:
        
        unexpected = received_domain_types.difference(expected_domain_types)
        if unexpected:
            msg = f'Unexpected domain types: {unexpected}.'
            raise ValueError(msg)
            
        unused = expected_domain_types.difference(received_domain_types)
        if unused:
            msg = f'Unused domain types: {unused}.'
            raise ValueError(msg)
    
    
    def _reindex_domains(self, user_domain_types):
        grid               = self.grid 
        voxel2domain_type  = self.voxel2domain_type
        shape2domain       = self.shape2domain
        domain2shape       = shape2domain.inverse
        voxel2shape        = self.voxel2shape
        voxel2domain       = self.voxel2domain
        domain_types       = user_domain_types
        domain2domain_type = self.domain2domain_type
        shape2voxel        = voxel2shape.inverse
        
        for new_domain_type, domains in domain_types.items():
            for new_domain, internal_point in domains.items():
                
                if not grid.contains(internal_point):
                    msg = f'Internal point {internal_point} appears to be outside the grid.'
                    raise ValueError(msg)
                
                #Determine the corresponding domain and domain_type  
                voxelized_point = grid.voxelize(internal_point)
                old_domain_type = voxel2domain_type[voxelized_point]
                
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
                
                #Reindex
                shape      = voxel2shape[voxelized_point]
                old_domain = shape2domain[shape]

                shapes2reindex = list(domain2shape[old_domain])
                for shape in shapes2reindex:
                    voxels     = shape2voxel[shape]
                    for voxel in voxels:
                        voxel2domain[voxel] = new_domain
                    shape2domain[shape] = new_domain                
                domain2domain_type[new_domain] = new_domain_type

    def _add_adjacent_domains(self, user_adjacent_domains):
        adjacent_shapes = self.adjacent_shapes
        shape2domain    = self.shape2domain

        shape2domain     = self.shape2domain
        adjacent_domains = self.adjacent_domains
        
        expected = {k: tuple(sorted(v)) for k, v in user_adjacent_domains.items()}
        expected = One2One('interface', 'domains', expected).inverse

        for pair, (shape0_voxels, shape1_voxels) in adjacent_shapes.items():
            shape0, shape1 = pair
            domain0        = shape2domain[shape0]
            domain1        = shape2domain[shape1]
            domains        = domain0, domain1
            domains_       = tuple(sorted(domains))
            
            #Generate the value
            if domains == domains_:
                value = AdjacentDomains(domain0, 
                                        domain1,
                                        shape0_voxels, 
                                        shape1_voxels
                                        )
            else:
                value = AdjacentDomains(domain1, 
                                        domain0,
                                        shape1_voxels, 
                                        shape0_voxels
                                        )
                
            #Generate the key
            key = expected.get(domains_)
            
            if key is None:
                msg = f'Detected an unexpected pair of adjacent domains {domains} not in the user input.'
                raise ValueError(msg)
            
            #Update self.adjacent_domains    
            adjacent_domains[key] = value
        
        #Check for missing
        missing = set(expected) - set(adjacent_domains)
        
        if missing:
            msg = f'Could not find the adjacent domains {missing} which were expected based on the user input.'
        
    def _add_adjacent_shapes(self, voxel, neighbour):
        voxel2shape     = self.voxel2shape
        adjacent_shapes = self.adjacent_shapes
        
        shape0       = voxel2shape[voxel]
        shape1       = voxel2shape[neighbour]
        
        
        key = tuple(sorted([shape0, shape1]))
        
        #Prevent double computation
        if key != (shape0, shape1):
            return
        
        default = make_adjacent_shapes()
        adjacent_shapes.setdefault(key, default)
        
        adjacent_shapes[key].shape0_voxels.append(voxel)
        adjacent_shapes[key].shape1_voxels.append(neighbour)
                
    def plot_voxels(self, 
                    ax, 
                    domain_args      = None,
                    **kwargs
                    ) -> tuple:
        
        #Parse plot args
        converters   = {'facecolor'   : upp.get_color, 
                        'surfacecolor': upp.get_color,
                        'color'       : upp.get_color
                        }
        
        shape_results, boundary_results, surface_results = super().plot_voxels(ax, 
                                                                               **kwargs
                                                                               )
        domain_results = []
        
        default_domain_args = self.default_domain_args
        for domain_type, domains in self.domain_types.items():
            for domain, internal_point in domains.items():
                
                domain_args_ = upp.process_kwargs(domain_args,
                                                    [domain_type],
                                                    default_domain_args,
                                                    {},
                                                    converters
                                                    )
                
                temp =  ax.text(*internal_point, domain, **domain_args_)   
                domain_results.append(temp)
        
        
        return (shape_results, boundary_results, surface_results, domain_results)
            