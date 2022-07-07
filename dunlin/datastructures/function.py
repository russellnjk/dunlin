from typing import Sequence

import dunlin.utils                       as ut
import dunlin.standardfile.dunl.writedunl as wd
from dunlin.datastructures.bases import _ADict, _AItem

class Function(_AItem):
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, ext_namespace: set, name: str, *items: Sequence[str]):
        #Extract args and expr
        *args, expr = items
        
        #Check
        [ut.check_valid_name(a) for a in args]
        
        expr_ori       = self.format_primitive(expr)
        expr           = str(expr).strip()
        expr_namespace = ut.get_namespace(expr)
        args_namespace = ut.get_namespace(args)
        
        undefined = expr_namespace.difference(args_namespace)
        if undefined:
            msg = f'Undefined namespace in function {name}: {undefined}'
            raise NameError(msg)
        
        #It is now safe to call the parent's init
        super().__init__(ext_namespace, name)
        
        #Save attributes
        self.name      = name
        self.expr      = expr
        self.expr_ori  = expr_ori
        self.args      = tuple(args)
        self.namespace = tuple(args_namespace)
        self.signature = ', '.join(self.args)
        
        #Check name and freeze
        self.freeze()
    
    ###########################################################################
    #Export
    ###########################################################################
    def to_data(self) -> list:
        return [*self.args, self.expr_ori]

class FunctionDict(_ADict):
    itype = Function
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, functions: dict, ext_namespace: set) -> None:
        #Make the dict
        super().__init__(functions, ext_namespace)
        
        #Freeze
        self.freeze()
    
    


