from numbers import Number
from typing  import KeysView, ValuesView, ItemsView, Iterable

import dunlin.utils as ut
from .bases               import DataDict, DataValue
from .coordinatecomponent import CoordinateComponentDict
from .compartment         import CompartmentDict

class Domain(DataValue):
    '''
    Differences with SBML Spatial:
        1. Domain types and compartments have been merged as explained in the 
        documentation for compartments. In dunlin, this class now takes over 
        some of the information handled by SBML domain types.
        
        2. Only one internal point is accepted. This avoids unecessary checking.
        
        3. Does not implement SpatialSymbolReference for domain types. This is 
        not something that should change with time so as to avoid unecessary 
        complexity.
    '''
    def __init__(self,
                 all_names             : set, 
                 coordinate_components : CoordinateComponentDict,
                 compartments          : CompartmentDict,
                 domain2compartment    : dict[str, str],
                 internal_points       : set[tuple],
                 name                  : str,
                 compartment           : str,
                 internal_point        : list[Number]
                 ) -> None:
        
        #Check name
        if not ut.is_valid_name(name):
            msg = f'Invalid name provieded for {type(self).__name__}: {name}'
            raise ValueError(msg)
        
        #Check compartment
        if compartment not in compartments:
            msg = f'Domain {name} was assigned to an undefined compartment: {compartment}.'
            raise ValueError(msg)
        
        domain2compartment[name] = compartment
        
        #Parse and check internal point
        spans = list(coordinate_components.spans.values())
        ndims = coordinate_components.ndims
        if len(internal_point) != coordinate_components.ndims:
            msg  = f'Expected an internal point with {ndims} coordinates.'
            msg += f' Received {internal_point}'
            raise ValueError(msg)
        
        for i, span in zip(internal_point, spans):
            if i <= span[0] or i >= span[1]:
                msg  = 'Internal point must be lie inside coordinate components.'
                msg += ' Spans: {spans}. Received {internal_point}'
                raise ValueError(msg)
        
        if tuple(internal_point) in internal_points:
            msg = f'Repeated internal points {internal_point}.'
            raise ValueError(msg)
            
        #Call the parent constructor
        super().__init__(all_names, 
                         name, 
                         compartment    = compartment, 
                         internal_point = tuple(internal_point)
                         )

    def to_dict(self) -> dict:
        dct = {'compartment'    : self.compartment,
               'internal_point' : list(self.internal_point)
               }
        dct = {self.name: dct}
        return dct

class DomainDict(DataDict):
    itype = Domain
    
    def __init__(self, 
                 all_names             : set,
                 coordinate_components : CoordinateComponentDict,
                 compartments          : CompartmentDict,
                 mapping               : dict
                 ):
        
        domain2compartment = {}
        internal_points    = set()
        super().__init__(all_names,
                         mapping, 
                         coordinate_components, 
                         compartments, 
                         domain2compartment,
                         internal_points
                         )
        
        
        