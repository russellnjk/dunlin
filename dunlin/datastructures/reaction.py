import re
from numbers import Number
from typing import Union

import dunlin.utils as ut
from dunlin.datastructures.bases import DataDict, DataValue
from dunlin.datastructures.stateparam import StateDict

class Reaction(DataValue):
    ###########################################################################
    #Preprocessing
    ###########################################################################
    @staticmethod
    def equation2stoich(equation: str) -> tuple[dict, set, set]:
        #An example equation is a + 2*b -> c
        #Split the reaction
        try:
            lhs, rhs = equation.split('->')
        except:
            msg = 'Invalid reaction. The expected format is <reactants> -> <products>.'
            msg = f'{msg} Received: {equation}'
            raise ValueError(msg)
        
        stoichiometry = {}
        reactants     = set()
        products      = set()
        
        def repl(match, is_positive):
            coefficient = ut.str2num(match[1])
            state       = match[2]
            
            if is_positive:
                stoichiometry[state] = coefficient
                reactants.add(state)
            else:
                stoichiometry[state] = -coefficient
                products.add(state)
        
        pattern = '[0-9]*\**([a-zA-Z]\w*)'
        re.match(pattern, lhs)
        re.match(pattern, rhs)
        return stoichiometry, reactants, products
       
    @staticmethod
    def get_rxn_rate(rate: Union[str, Number]) -> tuple[str, set]:
        rate = str(rate).strip()
        
        if not rate:
            msg = 'Invalid reaction rate. .'
            msg = f'{msg} Received: {rate}'
            raise ValueError(msg)
        
        namespace = ut.get_namespace(rate)
        
        return rate, namespace
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, 
                 all_names     : set, 
                 states        : StateDict, 
                 states_set    : set,
                 name          : str, 
                 stoichiometry : dict[str, Number],
                 rate          : str, 
                 bounds        : list[Number, Number]=None,
                 ) -> None:
            
        reactants      = set()
        products       = set()
        stoichiometry_ = {}
        
        for state, coefficient in stoichiometry.items():
            
            if not isinstance(coefficient, Number):
                msg = f'Reaction {name} contains a non-numeric stoichiometric coefficient for state {state}: {coefficient}.'
                raise TypeError(msg)
            
            stoichiometry_[state] = coefficient
            
            if coefficient > 0:
                reactants.add(state)
            else:
                products.add(state)
                
        #Parse the reaction rates
        rate, rxn_namespace = self.get_rxn_rate(rate)
        
        #Collect the namespace
        rxn_namespace.update(stoichiometry)
        
        #Check namespaces
        undefined = rxn_namespace.difference(all_names)
        if undefined:
            raise NameError(f'Undefined namespace: {undefined}.')
        
        all_states = states.names
        difference = set(stoichiometry).difference(all_states)
        if difference:
            msg = f'Stoichiometry for {name} contains unexpected states: {difference}.'
            raise ValueError(msg)
        
        #Parse the bounds
        bounds_ = None if bounds is None else tuple(bounds)
        
        #It is now safe to call the parent's init
        super().__init__(all_names, 
                         name, 
                         stoichiometry = stoichiometry_,
                         rate          = str(rate),
                         reactants     = frozenset(reactants),
                         products      = frozenset(products),
                         states        = frozenset(reactants|products),
                         bounds        = bounds_
                         )
        
        states_set.update(stoichiometry)
        
    ###########################################################################
    #Export
    ###########################################################################
    def to_dict(self) -> dict:
        dct = {'stoichiometry' : self.stoichiometry,
               'rate'          : ut.try2num(self.rate)
               }
        
        if self.bounds:
            dct['bounds'] = list(self.bounds)
            
        dct = {self.name: dct}
        return dct
    
class ReactionDict(DataDict):  
    itype = Reaction
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, 
                 all_names : set, 
                 states    : StateDict,
                 reactions : dict
                 ) -> None:
        
        states_set = set()
        
        #Make the dict
        super().__init__(all_names, reactions, states, states_set)
        
        #Save attributes
        self.states    = frozenset(states_set)
        