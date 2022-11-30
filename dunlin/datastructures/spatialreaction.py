from dunlin.datastructures.reaction import Reaction, ReactionDict

class SpatialReaction(Reaction):
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, 
                 ext_namespace: set, 
                 name         : str, 
                 eqn          : str, 
                 fwd          : str, 
                 rev          : str=None,
                 local        : bool=False
                 ) -> None:
        
        #Call the parent constructor without freezing
        super().__init__(ext_namespace, 
                         name, 
                         eqn, 
                         fwd, 
                         rev 
                         )
        #Unfreeze
        self.unfreeze()
        
        self.local =  local
        
        #Freeze
        self.freeze()
    
    def to_data(self) -> dict:
        data = super().to_data()
        
        if self.local:
            data['local'] = True
        
        return data

class SpatialReactionDict(ReactionDict):
    itype = SpatialReaction
    