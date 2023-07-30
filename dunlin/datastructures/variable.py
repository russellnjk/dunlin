from typing import Union

import dunlin.utils as ut
from dunlin.datastructures.bases import DataDict, DataValue

class Variable(DataValue):
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, 
                 all_names : set, 
                 name      : str, 
                 expr      : Union[str, int, float]
                 ) -> None:
        #Parse expression and check
        expr_str  = self.primitive2string(expr)
        namespace = ut.get_namespace(expr_str, allow_reserved=True)
        undefined = namespace.difference(all_names)
        if undefined:
            raise NameError(f'Undefined namespace: {undefined}.')
        
        #It is now safe to call the parent's init
        super().__init__(all_names, 
                         name      = name,
                         expr      = expr_str,
                         expr_ori  = expr,
                         namespace = namespace
                         )
    
    ###########################################################################
    #Export
    ###########################################################################
    def to_dict(self) -> dict:
        dct = {self.name: self.expr_ori}
        return dct

class VariableDict(DataDict):
    itype = Variable
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, 
                 all_names : set, 
                 variables : dict,
                 ) -> None:
        #Make the dict
        super().__init__(all_names, variables)
        