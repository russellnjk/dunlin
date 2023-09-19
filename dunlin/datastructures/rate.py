from numbers import Number
from typing  import Union

import dunlin.utils as ut
from dunlin.datastructures.bases import DataDict, DataValue
from .stateparam                 import StateDict

class Rate(DataValue):
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, 
                 all_names : set, 
                 states    : StateDict,
                 state     : str, 
                 expr      : Union[str, Number]
                 ):
        
        #Check that the state is in all_names
        if state not in states:
            raise NameError(f'Encountered a rate for an undefined state: {state}.')
        
        #Parse expression and check
        expr_str  = self.primitive2string(expr)
        namespace = ut.get_namespace(expr_str)
        undefined = namespace.difference(all_names)
        if undefined:
            msg = f'Encountered undefined items in rate for {state}: {undefined}.'
            raise NameError(msg)
            
        #Save attributes
        super().__init__(all_names, 
                         name      = None, 
                         expr      = expr_str, 
                         expr_ori  = expr, 
                         state     = state,
                         namespace = namespace
                         )
    
    def __str__(self) -> str:
        return f'{type(self).__name__}({repr(self.state)})'
    
    ###########################################################################
    #Export
    ###########################################################################
    def to_dict(self) -> dict:
        dct = {self.state: self.expr_ori}
        return dct
    
class RateDict(DataDict):
    itype = Rate
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, 
                 all_names : set, 
                 states    : StateDict, 
                 rates     : dict
                 ) -> None:
        #Make the dict
        super().__init__(all_names, rates, states)
       
        #Save attributes
        self.states = frozenset(self.keys())
        