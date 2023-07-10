import re

from dunlin.datastructures.bases       import DataDict
from dunlin.datastructures.reaction    import Reaction, ReactionDict
from dunlin.datastructures.compartment import CompartmentDict

class SpatialReaction(Reaction):
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, 
                 all_names    : set,
                 compartments : CompartmentDict,
                 name         : str, 
                 eqn          : str, 
                 fwd          : str, 
                 rev          : str=None,
                 local        : bool=False
                 ):
        
        #Call the parent constructor and unfreeze
        super().__init__(all_names, 
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
        
    def to_dict(self) -> dict:
        dct = super().to_data()
        
        if self.local:
            dct[self.name]['local'] = True
        
        return dct

class SpatialReactionDict(DataDict):
    itype = SpatialReaction
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, 
                 all_names    : set, 
                 compartments : CompartmentDict,
                 reactions    : dict
                 ) -> None:
        
        #Make the dict
        super().__init__(all_names, reactions, compartments)
        
        