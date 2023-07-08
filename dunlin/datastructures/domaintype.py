from typing import KeysView, ValuesView, ItemsView, Iterable

import dunlin.utils as ut
from .bases               import DataDict, DataValue
from .coordinatecomponent import CoordinateComponentDict

class DomainType(DataValue):
    '''
    Differences with SBML Spatial:
        1. Merged with SBML Compartments. Compartments represent physical 
        locations of chemical species. In other words, species in different 
        compartments are physically separated and should not occur in the 
        same region. 
        In SBML, the geometry is defined entirely be domain types which correspond 
        to areas where its associated species have the same initial value. If 
        
        
        SBML allows a one-to-one mapping between species and compartments, 
        and a many-to-one mapping between compartments and domain types.
        This is confusing because the geometry is defined entirely be domain 
        types. The result is that the compartments cannot be mapped to a specific 
        region of the geometry. Instead, the region specified by a domain type 
        will contain all its associated species throughout its entirety. This 
        contradicts the purpose of compartments which is to allow different 
        species to be separated.
        
        Also, to allow a domain type to be differentiated into compartments, SBML 
        has each compartment define an attribute called the unit size. There 
        are two ways to interpret the unit size:
            1. The unit sizes sum to one. Each compartment represents a fraction 
            of domain type although where exactly each compartment exists is 
            left undefined in the model.
            2. The unit size represents a conversion fact e.g. 3D to 2D.
        
        This is problematic because there is no way to know beforehand which 
        interpretation to use. And if the second interpretation is used, it is 
        impossible to know what kind of conversion is intended by the modeller.
        
        This results in unecessary confusion and ambiguity so I propose merging 
        compartments and domain types. States have a one-to-one mapping with 
        compartments. The resulting datastructures are much simpler and easier 
        to understand.
        
        1. Does not accept a parameter for number of dimensions i.e. 
        spatialDimensions. The number of dimensions is inferred from the internal 
        point. 
        2. Only one internal point is accepted. This avoids unecessary checking.
        3. Does not implement SpatialSymbolReference. This is not something that 
        should change with time so as to avoid unecessary complexity.
    '''
    def __init__(self,
                 all_names: set, 
                 coordinate_components: CoordinateComponentDict,
                 name: str,
                 **domains: dict[str, list],
                 ) -> None:
        
        #Check name
        if not ut.is_valid_name(name):
            msg = f'Invalid name provieded for {type(self).__name__}: {name}'
            raise ValueError(msg)
        
        #Infer ndims
        ndims = coordinate_components.ndims
        
        #Check domains
        spans    = list(coordinate_components.spans.values())
        _domains = {}
        for domain_name, internal_point in domains.items():
            if not ut.is_valid_name(domain_name):
                msg  = 'Invalid domain name provieded for '
                msg += f'{type(self).__name__}: {domain_name}'
                raise ValueError(msg)
            
            _domains[domain_name] = []
            
            if len(internal_point) != coordinate_components.ndims:
                msg  = f'Expected an internal point with {ndims} coordinates.'
                msg += f' Received {internal_point}'
                raise ValueError(msg)
            
            for i, span in zip(internal_point, spans):
                if i <= span[0] or i >= span[1]:
                    msg  = 'Internal point must be lie inside coordinate components.'
                    msg += ' Spans: {spans}. Received {internal_point}'
                    raise ValueError(msg)
                
            _domains[domain_name] = tuple(internal_point)
        
        #Call the parent constructor
        super().__init__(all_names, name, _domains=_domains, ndims=ndims)
        
    @property
    def domains(self) -> dict:
        return self._domains

    def to_dict(self) -> dict:
        d   = {k: list(v) for k, v in self.domains.items()}
        dct = {self.name: d}
        return dct
    
    def keys(self) -> KeysView:
        return self._domains.keys()
    
    def values(self) -> ValuesView:
        return self._domains.values()
    
    def items(self) -> ItemsView:
        return self._domains.items()
    
    def __iter__(self) -> Iterable:
        return iter(self._domains)

class DomainTypeDict(DataDict):
    itype = DomainType
    
    def __init__(self, 
                 all_names: set,
                 coordinate_components: CoordinateComponentDict,
                 mapping: dict
                 ) -> None:
        
        super().__init__(all_names, mapping, coordinate_components)
        
        seen_domains         = set()
        seen_internal_points = set()
        for domain_type in self.values():
            for domain, internal_points in domain_type.domains.items():
                #Check for repeated domain names
                if domain in seen_domains:
                    msg = f'Redefinition of domain {domain}.'
                    raise ValueError(msg)
                
                #Check for repeated points
                repeated = seen_internal_points.intersection(internal_points)
                if repeated:
                    msg = f'Repeated internal points {repeated}.'
                    raise ValueError(msg)
                
                seen_domains.add(domain)
                seen_internal_points.update(internal_points)
                
    @property
    def domains(self) -> list:
        lst = []
        for domain_type in self.values():
            lst += list(domain_type)
        
        return lst
        
        