from numbers import Number
from typing  import Literal, Union

import dunlin.utils as ut
from .bases               import GenericItem, GenericDict
from .coordinatecomponent import CoordinateComponentDict
from .domaintype          import DomainTypeDict

class BoundaryCondition(GenericItem):
    def __init__(self,
                 ext_namespace: set(),
                 domain_types: DomainTypeDict,
                 name: str,
                 state: str,
                 condition: Number,
                 condition_type: Literal['Neumann', 'Dirichlet', 'Robin'],
                 domain_type: str
                 ) -> None:
        
        #Check condition_type
        allowed = ['Neumann', 'Dirichlet', 'Robin']
        if condition_type not in allowed:
            msg  = f'Invalid condition type in {name}: {condition_type}.'
            msg += f'Expected one of {allowed}.'
            raise ValueError(msg)
        
        #Parse and check the condition
        if condition_type == 'Robin':
            raise NotImplementedError('Robin condition not implemented yet.')
        else:
            if not ut.isnum(condition):
                msg  = f'Expected a number for boundary condition {name}. '
                msg += 'Received condition of type {type(condition).__name__}.'
                raise TypeError(msg)

        #Check the domain_type
        if domain_type not in domain_types:
            msg = f'Unexpected domain_type in {name}: {domain_type}'
            raise ValueError(msg)
        
        #Check state
        if not ut.is_valid_name(state):
            msg = f'Invalid state name {state} in {name}.'
            raise ValueError(msg)
        
        #Call the parent constructor            
        super().__init__(ext_namespace, 
                         name, 
                         state=state,
                         condition_type=condition_type,
                         condition=condition,
                         domain_type=domain_type
                         )
        
        #Freeze
        self.freeze()
    
    def to_data(self) -> list:
        lst = [self.state, 
               self.condition, 
               self.condition_type, 
               self.domain_type
               ]
        return lst
    
class BoundaryConditionDict(GenericDict):
    itype = BoundaryCondition
    
    def __init__(self,
                 ext_namespace: set,
                 domain_types: DomainTypeDict,
                 mapping: dict
                 ) -> None:
        super().__init__(ext_namespace, mapping, domain_types)
        
        seen = set()
        
        for bc in self.values():
            state     = bc.state
            domain_type = bc.domain_type
            temp        = state, domain_type
            
            if temp in seen:
                msg = f'Found multiple boundary conditions for {temp}.'
                raise ValueError(msg)
            
            seen.add(temp)
            