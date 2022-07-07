from typing import Union

import dunlin.utils                       as ut
import dunlin.standardfile.dunl.writedunl as wd
from dunlin.datastructures.bases import _ADict, _AItem

class Rate(_AItem):
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, ext_namespace: set, name: str, expr: Union[str, int, float]):
        #Check that the state is in ext_namespace
        if name not in ext_namespace:
            raise NameError(f'Encountered a Rate for an undefined namespace: {name}')
        
        #Parse expression and check
        expr_ori  = self.format_primitive(expr)
        expr      = str(expr).strip()
        namespace = ut.get_namespace(expr)
        undefined = namespace.difference(ext_namespace)
        if undefined:
            raise NameError(f'Undefined namespace: {undefined}.')
            
        #It is now safe to call the parent's init
        #Use the derivate, not the state name
        d_name = ut.diff(name)
        super().__init__(ext_namespace, name, d_name)
        
        
        #Save attributes
        self.expr      = expr
        self.expr_ori  = expr_ori
        self.namespace = tuple(namespace)
        self.state     = name
        
        #Freeze
        self.freeze()
    
    ###########################################################################
    #Export
    ###########################################################################
    def to_data(self) -> str:
        return self.expr_ori
    

class RateDict(_ADict):
    itype = Rate
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, rates: dict, ext_namespace: set) -> None:
        namespace = set()
        
        def callback(name, value):
            namespace.update(value.namespace)

        #Make the dict
        super().__init__(rates, ext_namespace, callback)
        
        #Save attributes
        self.namespace = tuple(namespace)
        
        #Freeze
        self.freeze()

