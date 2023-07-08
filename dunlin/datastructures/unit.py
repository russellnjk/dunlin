from .bases      import Table
from .stateparam import StateDict, ParameterDict

class UnitsDict(Table):
    itype      = 'Units'
    is_numeric = False
    
    def __init__(self, 
                 states     : StateDict,
                 parameters : ParameterDict,
                 mapping    : dict
                 ):
        if mapping:
            #If mapping is provided, make sure all states and parameters have units
            expected = set(states.keys()) | set(parameters.keys())
            missing  = expected.difference(mapping)
            
            if missing:
                msg = f'Missing units for {missing}.'
                raise ValueError(msg)
            
            unexpected = set(mapping).difference(expected)
            
            if unexpected:
                msg = f'Units for unexpected items: {unexpected}.'
                raise ValueError(msg)
            
        super().__init__(set(), mapping)
    
    def to_dict(self) -> dict:
        return self.df.iloc[0].to_dict()