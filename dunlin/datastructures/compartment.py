from numbers import Number

import dunlin.utils                    as ut
import dunlin.datastructures.exception as exc
from dunlin.datastructures.bases import DataDict, DataValue
from .stateparam                 import StateDict

class Compartment(DataValue):
    '''
    Differences with SBML Spatial:
        1. Merged with domain types. Compartments represent physical 
        locations of chemical species. Meanwhile, domain types represent regions 
        in space where a particular species exists and has a particular initial 
        value.
        
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
        
        2. Does not accept a parameter for number of dimensions i.e. 
        spatialDimensions. The number of dimensions is inferred from the internal 
        point. 
        
        3. Does not implement SpatialSymbolReference for domain types. This is 
        not something that should change with time so as to avoid unecessary 
        complexity.
    '''
    def __init__(self,
                 all_names         : set,
                 all_states        : StateDict,
                 state2compartment : dict[str, str],
                 name              : str,
                 *states           : str,
                 ):
        
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
                
        #Call the parent constructor
        super().__init__(all_names, 
                         name, 
                         states = frozenset(states),
                         )
        
        
    def to_dict(self) -> dict:
        dct = {self.name: list(self.states)}
        return dct

class CompartmentDict(DataDict):
    itype = Compartment
    
    def __init__(self, 
                 all_names : set, 
                 states    : StateDict,
                 mapping   : dict
                 ) -> None:
        
        state2compartment = {}
        
        super().__init__(all_names, 
                         mapping, 
                         states, 
                         state2compartment,
                         )
        
        
        missing = set(states.names).difference(state2compartment)
        if missing:
            msg = f'Compartments not assigned for {missing}.'
            raise ValueError(msg)
        
        #Update
        self.state2compartment       = state2compartment
        