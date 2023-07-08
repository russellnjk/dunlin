from numbers import Number

import dunlin.utils                    as ut
import dunlin.datastructures.exception as exc
from dunlin.datastructures.bases import DataDict, DataValue
from .stateparam                 import StateDict
from .reaction                   import ReactionDict 
from .domaintype                 import DomainTypeDict

class Compartment(DataValue):
    def __init__(self,
                 all_names               : set,
                 all_states              : StateDict,
                 domain_types            : DomainTypeDict,
                 state2compartment       : dict[str, str],
                 compartment2domain_type : dict[str, str],
                 state2domain_type       : dict[str, str],
                 name                    : str,
                 domain_type             : str,
                 states                  : list[str],
                 unit_size               : Number         = 1,
                 ):
        
        #Check the domain_type. Update if valid.
        if domain_type not in domain_types:
            msg  = f'Compartment {name} was assigned to an undefined domain type: {domain_type}.'
            raise NameError(msg)
        
        compartment2domain_type[name] = domain_type
        
        #Check the states
        for state in states:
            if state not in all_states:
                msg = f'Compartment {name} contains an undefined state: {state}.'
                raise NameError(msg)
            elif state in state2compartment:
                msg = f'State {state} was assigned to at least two compartments:'
                msg = f'{msg} {name} and {state2compartment[state]}.'
                raise ValueError(msg)
            
            state2compartment[state] = name
            state2domain_type[state] = domain_type
                
        #Check the unit size
        if not ut.isnum(unit_size):
            msg = f'Unit size in {name} must be a number.'
            raise ValueError(msg)
        
        #Call the parent constructor
        super().__init__(all_names, 
                         name, 
                         domain_type = domain_type,
                         states      = list(states),
                         unit_size   = unit_size,
                         )
        
        
        
    def to_dict(self) -> dict:
        dct = {'domain_type' : self.domain_type,
               'states'      : list(self.namespace),
               }
        if self.unit_size != 1:
            dct['unit_size'] = self.unit_size
        
        dct = {self.name: dct}
        return dct

class CompartmentDict(DataDict):
    itype = Compartment
    
    def __init__(self, 
                 all_names: set, 
                 states       : StateDict,
                 domain_types : DomainTypeDict,
                 mapping      : dict
                 ) -> None:
        
        state2compartment       = {}
        domain_type2compartment = {}
        state2domain_type       = {}
        super().__init__(all_names, 
                         mapping, 
                         states, 
                         domain_types, 
                         state2compartment,
                         domain_type2compartment,
                         state2domain_type
                         )
        
        
        missing = set(states.names).difference(state2compartment)
        if missing:
            msg = f'Compartments not assigned for {missing}.'
            raise ValueError(msg)
        
        #Update
        self.state2compartment       = state2compartment
        self.state2domain_type       = state2domain_type
        self.domain_type2compartment = domain_type2compartment
        