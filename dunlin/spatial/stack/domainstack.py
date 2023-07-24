from collections import namedtuple
from numbers import Number
from typing  import Iterable, Mapping, Union

import dunlin.utils_plot as upp
from ..grid.grid   import RegularGrid, NestedGrid
from .bidict       import One2One, One2Many
from .stack        import (Stack,
                           Domain_type, Domain, Voxel,
                           )

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
    
    grids              : dict[str, Union[RegularGrid, NestedGrid]]
    adjacent_shapes    : set[tuple[str, str]]
    voxel2domain       : One2Many[Voxel, Domain]
    domain2domain_type : One2Many[Domain, Domain_type]
    shape2domain       : One2Many[str, Domain]
    surface2domain     : One2One[Union[str, tuple[Domain, Domain]], tuple[Domain, Domain]]
    
    def __init__(self, 
                 grid_config         : dict, 
                 shapes              : Iterable,
                 domain_type2domain  : dict[Domain_type, dict[Domain, tuple[Number]]],
                 surface2domain      : dict[str, list[Domain, Domain]] = None
                 ) -> None:
        
        #Generate the grids
        nested_grids = self.make_grids_from_config(grid_config)
        grid         = nested_grids['_main']
        self.grids   = nested_grids
        
        #Template the mappings for _add_voxel 
        self.adjacent_shapes = set()
        
        #Temporary attribute
        #Need to make it accessible to _preprocess
        self.domain_type2domain = domain_type2domain
        
        #Call the parent constructor
        super().__init__(grid, shapes)
        
        #Delete temporary attributes
        del self.domain_type2domain
        
        #Check that all voxels have been assigned to a domain
        assigned = set(self.voxel2domain)
        expected = set(self.voxels)
        missing  = expected.difference(assigned)
        
        if missing:
            msg = f'The following voxels could not be assigned to a domain: {missing}.'
            raise ValueError(msg)
            
        #Check surface2domain if it was provided
        #Create the attribute
        if surface2domain:
            self._check_surfaces(surface2domain)
            temp                = {k: tuple(v) for k, v in surface2domain.items()}
            self.surface2domain = One2One('surface', 'domain pair', temp)
        else:
            self.surface2domain = One2One('surface', 'domain pair')
            
    def _preprocess(self) -> None:
        super()._preprocess()
        
        #Make prelimnary mappings that will be modified later
        mappings = self._map_domains(self.grid, 
                                     self.voxel2domain_type_idx, 
                                     self.voxel2shape, 
                                     self.domain_type2domain
                                     )
        
        self.shape2domain       = mappings[0]
        self.domain2domain_type = mappings[1]
        self.domain2point       = mappings[2]
        
        #Make empty mappings that will be filled later
        self.voxel2domain   = One2Many('voxel', 'domain')
        self.surface2domain = One2Many('surface', 'domain')
        
    @staticmethod
    def _map_domains(grid, 
                     voxel2domain_type_idx : One2One,
                     voxel2shape           : One2Many, 
                     domain_type2domain    : dict,
                     ) -> tuple[One2Many, One2Many]:
        
        shapes             = list(voxel2shape.values())
        shape2domain       = {shape: i for i, shape in enumerate(shapes)}
        shape2domain       = One2Many('shape', 'domain', shape2domain)
        domain2domain_type = One2Many('domain', 'domain type')
        domain2point       = One2One('domain', 'point')
        
        for new_domain_type, domains in domain_type2domain.items():
            for domain, internal_point in domains.items():
                
                if not grid.contains(internal_point):
                    msg = f'Internal point {internal_point} appears to be outside the grid.'
                    raise ValueError(msg)
                
                #Convert the internal point into a voxel and get its domain type
                voxel           = grid.voxelize(internal_point)
                old_domain_type = voxel2domain_type_idx[voxel][1]
                
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
                
                shape2domain[shape] = domain
                
                #Update domain2domain_type
                domain2domain_type[domain] = new_domain_type
                
                #Update domain2point
                domain2point[domain] = tuple(internal_point)
                
        return shape2domain, domain2domain_type, domain2point
    
    def _check_surfaces(self, surface2domain: dict) -> None:
        
        domain2surface  = {tuple(sorted(v)): k for k, v in surface2domain.items()}
        shape2domain    = self.shape2domain
        
        for pair in self.adjacent_shapes:
            domain0 = shape2domain[pair[0]]
            domain1 = shape2domain[pair[1]]
            domains = tuple(sorted([domain0, domain1]))
            
            if domains not in domain2surface:
                msg  = 'A mapping of surfaces to domains was provided. '
                msg += 'However, no surface could be found for the domain pair {domains}.'
                raise ValueError(msg)
            
    def _reindex_domains(self, domain_type2domain: dict) -> None:
        grid                   = self.grid 
        voxel2domain_type_idx  = self.voxel2domain_type_idx
        voxel2domain_type      = self.voxel2domain_type
        voxel2shape            = self.voxel2shape
        shape2voxel            = voxel2shape.inverse
        old_shape2domain       = self.shape2domain
        new_shape2domain       = One2Many('shape', 'domain')
        voxel2domain           = self.voxel2domain
        domain2domain_type_idx = self.domain2domain_type_idx
        missing                = set(self.shape2domain.values())
        
        for new_domain_type, domains in domain_type2domain.items():
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
    
    def _add_voxel(self, voxel) -> None:
        super()._add_voxel(voxel)
        
        voxel2shape  = self.voxel2shape
        shape2domain = self.shape2domain
        voxel2domain = self.voxel2domain
        
        shape  = voxel2shape[voxel]
        domain = shape2domain.get(shape)
        if domain is not None:
            voxel2domain[voxel] = domain
        
    def _add_bulk(self, voxel, neighbour, shift) -> None:
        voxel2shape  = self.voxel2shape
        shape2domain = self.shape2domain
        
        shape0 = voxel2shape[voxel]
        shape1 = voxel2shape[neighbour]
        
        domain0 = shape2domain[shape0]
        domain1 = shape2domain[shape1]
        
        #Update shape2domain
        #Case 1: Both domains are integers
        #Reassign one of them to the other. Doesn't matter which one.
        if type(domain0) == int and type(domain1) == int:
            shape2domain[shape1] = domain0
        #Case 2: Domain 1 is an integer but domain 0 isn't.
        #Reassign domain 1
        elif type(domain1) == int:
            shape2domain[shape1] = domain0
        #Case 3: Domain 0 is an integer but domain 1 isn't.
        #Reassign domain 0
        elif type(domain0) == int:
            shape2domain[shape0] = domain1
        #Case 4: Neither domain is an integer but they are the same.
        #No action required.
        elif domain0 == domain1:
            pass
        #Case 5: Neither domain is an integer but they are different.
        #This means two user domains of the same domain type are touching.
        #Raise an exception.
        else:
            msg  = f'Domains {domain0} and {domain1} are touching. '
            msg += 'Domains of the same domain type must not touch.'
            raise ValueError(msg)

    def _add_surface(self, voxel, neighbour, shift) -> None:
        voxel2shape     = self.voxel2shape
        adjacent_shapes = self.adjacent_shapes
        
        shape0 = voxel2shape[voxel]
        shape1 = voxel2shape[neighbour]
        
        pair = tuple(sorted([shape0, shape1]))
        
        adjacent_shapes.add(pair)
        
    def plot_voxels(self, 
                    ax, 
                    label_domains              : bool = False,
                    domain_textsize            : Number = 12,
                    domain_horizontalalignment : str = 'center',
                    **kwargs
                    ) -> tuple:
        results = super().plot_voxels(ax, **kwargs)
        
        domain_results = []
        for domain, point in self.domain2point.items():
            
            temp = ax.text(*point, 
                           domain, 
                           size=domain_textsize, 
                           horizontalalignment=domain_horizontalalignment
                           )
            
            domain_results.append(temp)
        
        results = results + (domain_results, )
        
        return results
            