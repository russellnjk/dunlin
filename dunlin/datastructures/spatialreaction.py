import re

from dunlin.datastructures.bases       import NamespaceDict
from dunlin.datastructures.reaction    import Reaction
from dunlin.datastructures.compartment import CompartmentDict

class SpatialReaction(Reaction):
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, 
                 ext_namespace: set,
                 compartments : CompartmentDict,
                 name         : str, 
                 eqn          : str, 
                 fwd          : str, 
                 rev          : str=None,
                 local        : bool=False
                 ) -> None:
        
        #Call the parent constructor and unfreeze
        super().__init__(ext_namespace, 
                         name, 
                         eqn, 
                         fwd, 
                         rev 
                         )
        self.unfreeze()
        
        #Continue pre-processing
        domain_type2state = {}

        for x in self.stoichiometry:
            dmnt = compartments.state2domain_type[x]
            domain_type2state.setdefault(dmnt, set()).add(x)
            
        if len(domain_type2state) > 2:
            msg = f'Encountered reaction {name} with multiple domain_types.'
            d   = '\n'.join([f'{k}: {v}' for k, v in domain_type2state.items()])
            msg = f'{msg}\n{d}'
            raise ValueError(msg)
            
        #Update attributes
        self.domain_type2state = domain_type2state
        self.local             = local
        
        #Freeze
        self.freeze()
    
    def to_data(self) -> dict:
        data = super().to_data()
        
        if self.local:
            data['local'] = True
        
        return data

class SpatialReactionDict(NamespaceDict):
    itype = SpatialReaction
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, 
                 ext_namespace : set, 
                 compartments  : CompartmentDict,
                 reactions     : dict
                 ) -> None:
        namespace = set()
        
        #Make the dict
        super().__init__(ext_namespace, reactions, compartments)
        
        states = set()
        
        for rxn_name, rxn in self.items():
            namespace.update(rxn.namespace)
            states.update(list(rxn.stoichiometry))
           
        #Save attributes
        self.namespace            = tuple(namespace)
        self.states               = tuple(states)
        
        #Freeze
        self.freeze()
    
    