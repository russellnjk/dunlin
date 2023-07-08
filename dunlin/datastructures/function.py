from typing import Sequence

import dunlin.utils as ut
from dunlin.datastructures.bases import DataDict, DataValue

class Function(DataValue):
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, all_names: set, name: str, *items: Sequence[str]):
        #Extract args and expr
        *args, expr = items
        
        #Check
        [ut.check_valid_name(a) for a in args]
        
        expr_str       = self.primitive2string(expr)
        expr_namespace = ut.get_namespace(expr)
        args_namespace = ut.get_namespace(args)
        
        undefined = expr_namespace.difference(args_namespace)
        if undefined:
            msg = f'Undefined namespace in function {name}: {undefined}'
            raise NameError(msg)
        
        #It is now safe to call the parent's init
        super().__init__(all_names, 
                         name,
                         expr      = expr_str,
                         expr_ori  = expr,
                         signature = tuple(args)
                         )
    ###########################################################################
    #Export
    ###########################################################################
    def to_dict(self) -> dict:
        lst = [*self.signature, self.expr_ori]
        dct = {self.name: lst}
        return dct

class FunctionDict(DataDict):
    itype = Function
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, all_names: set, functions: dict) -> None:
        #Make the dict
        super().__init__(all_names, functions)
        
