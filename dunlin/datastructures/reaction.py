import re
from numbers import Number
from typing import Union

import dunlin.utils as ut
from dunlin.datastructures.bases import DataDict, DataValue
from dunlin.datastructures.stateparam import StateDict

class Reaction(DataValue):
    '''
    This class differs from its SBML counterpart in several ways:
        1. Reversibility
        In SBML, reversibility is explicitly specified by an attribute. However, 
        this is unecessary as the reaction rate already contains this information; 
        it is assumed that the user provides appropriate parameter values which 
        lead to sensible reaction rate calculations.
        
        2. Local reactions
        In SBML spatial, the reaction with the isLocal attribute set to True must 
        also have a compartment attribute defined. Dunlin implements compartments 
        differently from SBML so these two attributes are not used. Reactions 
        that take places at surfaces (i.e. the boundary between two Dunlin 
        compartments) will have units of flux. In conjunction with the size of 
        the voxel, the flux will be used to be calculate the change in concentration 
        in that voxel.
        
    '''
    ###########################################################################
    #Preprocessing
    ###########################################################################
    @staticmethod
    def parse_equation(name: str, equation: str) -> tuple[dict, set, set]:
        #An example equation is -a -2*b +c
        msg = 'Equation parsing not implemented yet.'
        raise NotImplementedError(msg)
        
    @staticmethod
    def parse_stoichiometry(name: str, stoichiometry: dict) -> tuple[dict, set, set]:
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
                
        
        return stoichiometry_, reactants, products
    
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
        
        if type(stoichiometry) == dict:
            stoichiometry_, reactants, products = self.parse_stoichiometry(name, stoichiometry)
        elif type(stoichiometry) == str:
            stoichiometry_, reactants, products = self.parse_equation(name, stoichiometry)
        
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
        