from numbers import Number

import dunlin.utils                    as ut
import dunlin.datastructures.exception as exc
from dunlin.datastructures.bases import NamespaceDict, GenericItem
from .stateparam                 import StateDict
from .reaction                   import ReactionDict 
from .domaintype                 import DomainTypeDict

class Compartment(GenericItem):
    def __init__(self,
                 ext_namespace: set,
                 name         : str,
                 domain_type  : str,
                 contains     : list[str],
                 unit_size    : Number=1,
                 ) -> None:
        
        #Check the inputs
        if not ut.is_valid_name(domain_type):
            msg = f'Invalid domain type provided to {name}: {domain_type}'
            raise ValueError(msg)
        
        if type(contains) == str:
            contains = [contains]
        elif not ut.islistlike(contains):
            msg  = 'Expected a list-like container of strings. '
            msg += f'Received {contains} of type {type(contains).__name__}.'
            raise ValueError(msg)
            
        contains_ = []
        for item in contains:
            if not ut.is_valid_name(item):
                msg = f'Invalid name in {contains}.'
            else:
                contains_.append(str(item))
        
        if not ut.isnum(unit_size):
            msg = f'Unit size in {name} must be a number.'
            raise ValueError(msg)
        
        #Call the parent constructor
        super().__init__(ext_namespace, 
                         name, 
                         domain_type=domain_type,
                         namespace=tuple(contains_),
                         unit_size=unit_size,
                         )
        
        #Freeze
        self.freeze()
    
    def to_data(self) -> dict:
        dct = {'domain_type': self.domain_type,
               'contains'   : list(self.namespace),
               }
        if self.unit_size != 1:
            dct['unit_size'] = self.unit_size
        
        return dct

class CompartmentDict(NamespaceDict):
    itype = Compartment
    
    def __init__(self, 
                 ext_namespace: set, 
                 states       : StateDict,
                 domain_types : DomainTypeDict,
                 mapping      : dict
                 ) -> None:
        
        super().__init__(ext_namespace, mapping)
        
        allowed           = set(states.keys()) 
        seen              = set()
        namespace         = set()
        dmnt_names        = set()
        state2compartment = {}
        
        for cpt_name, cpt in self.items():
            #Check that the domain type has been defined
            domain_type = cpt.domain_type
            if domain_type not in domain_types:
                msg  = f'Compartment {cpt_name} requires a domain type '
                msg += f'"{cpt.domain_type}" which is undefined.'
                raise ValueError(msg)
            elif domain_type in dmnt_names:
                msg = f'Multiple compartments map to the domain_type {domain_type}.'
                raise ValueError(msg)
            else:
                dmnt_names.add(domain_type)
                
            #Check that all names are states or reactions
            temp = set(cpt.namespace)
            unexpected = temp.difference(allowed)
            if unexpected:
                msg  = f'Encountered unexpected names in {cpt_name}: {unexpected}.'
                msg += ' All names in the "contains" argument of a compartment '
                msg += 'must correspond to a state.'
                raise ValueError(msg)
            
            #Check that items do not appear in multiple compartments
            repeated = seen.intersection(temp)
            if repeated:
                msg = f'The following items appeared in multiple compartments: {repeated}.'
                raise ValueError(msg)
            
            #Update the namespace
            namespace.update(temp)
            
            #Update state2compartment 
            for state in cpt.namespace:
                state2compartment[state] = cpt
        
        missing = set(states.names).difference(namespace)
        if missing:
            msg = f'Compartments not assigned for {missing}.'
            raise ValueError(msg)
        
        #Update and freeze
        self.namespace          = tuple(namespace)
        self._state2compartment = state2compartment
        self.freeze()
            
        
    def locate_state(self, state) -> Compartment:
        return self._state2compartment[state]