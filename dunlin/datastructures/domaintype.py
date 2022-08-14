from typing import KeysView, ValuesView, ItemsView, Iterable

import dunlin.utils as ut
from .bases               import GenericItem, GenericDict
from .coordinatecomponent import CoordinateComponentDict

class DomainType(GenericItem):
    def __init__(self,
                 ext_namespace: set, 
                 coordinate_components: CoordinateComponentDict,
                 name: str,
                 domains: dict[str, list],
                 ndims: int=None
                 ) -> None:
        
        #Check name
        if not ut.is_valid_name(name):
            msg = f'Invalid name provieded for {type(self).__name__}: {name}'
            raise ValueError(msg)
        
        #Check ndims
        ndims = coordinate_components.ndims if ndims is None else ndims
        if not ut.isint(ndims):
            msg  = 'ndims must be a positive integer. '
            msg += f'Received {ndims}'
            raise ValueError(msg)
        elif ndims > 3 or ndims < 1:
            msg  = 'ndims must be between 1 and 3 inclusive. '
            msg += f'Received {ndims}'
            raise ValueError(msg)
        
        #Check domains
        spans    = list(coordinate_components.spans.values())
        _domains = {}
        for domain_name, internal_points in domains.items():
            if not ut.is_valid_name(domain_name):
                msg  = 'Invalid domain name provieded for '
                msg += f'{type(self).__name__}: {domain_name}'
                raise ValueError(msg)
            
            _domains[domain_name] = []
            
            for point in internal_points:
                if len(point) != coordinate_components.ndims:
                    msg  = f'Expected an internal point with {ndims} coordinates.'
                    msg += f' Received {point}'
                    raise ValueError(msg)
                
                for i, span in zip(point, spans):
                    if i <= span[0] or i >= span[1]:
                        msg  = 'Internal point must be lie inside coordinate components.'
                        msg += ' Spans: {spans}. Received {point}'
                        raise ValueError(msg)
                
            _domains[domain_name].append(tuple(point))
        
        #Call the parent constructor
        super().__init__(ext_namespace, name, _domains=_domains, ndims=ndims)
        
        #Freeze
        self.freeze()
    
    @property
    def domains(self) -> dict:
        return self._domains

    def to_data(self) -> dict:
        domains = {k: [list(i) for i in v] for k, v in self._domains.items()}
        ndims   = self.ndims
        
        return {'ndims': ndims, 'domains': domains}
    
    def keys(self) -> KeysView:
        return self._domains.keys()
    
    def values(self) -> ValuesView:
        return self._domains.values()
    
    def items(self) -> ItemsView:
        return self._domains.items()
    
    def __iter__(self) -> Iterable:
        return iter(self._domains)

class DomainTypeDict(GenericDict):
    itype = DomainType
    
    def __init__(self, 
                 ext_namespace: set,
                 coordinate_components: CoordinateComponentDict,
                 mapping: dict
                 ) -> None:
        
        super().__init__(ext_namespace, mapping, coordinate_components)
        
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
        
        