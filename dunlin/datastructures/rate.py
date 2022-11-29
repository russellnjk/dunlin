from typing import Union

import dunlin.utils                       as ut
from dunlin.datastructures.bases import NamespaceDict, GenericItem

class Rate(GenericItem):
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, ext_namespace: set, name: str, expr: Union[str, int, float]):
        #Check that the state is in ext_namespace
        if name not in ext_namespace:
            raise NameError(f'Encountered a Rate for an undefined namespace: {name}')
        
        ut.check_valid_name(name)
        
        #Parse expression and check
        expr_ori  = self.format_primitive(expr)
        expr      = str(expr).strip()
        namespace = ut.get_namespace(expr)
        undefined = namespace.difference(ext_namespace)
        if undefined:
            raise NameError(f'Undefined namespace: {undefined}.')
            
        #Use the derivate, not the state name as this object's name
        d_name = ut.diff(name)
        
        #Mimic the parent constructor
        self.name = d_name
        
        if d_name in ext_namespace:
            raise NameError(f'Redefinition of {d_name}.')
        else:
            #Update the namespace
            ext_namespace.add(d_name)
            
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
    
class RateDict(NamespaceDict):
    itype = Rate
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, ext_namespace: set, rates: dict) -> None:
        namespace = set()
        
        #Make the dict
        super().__init__(ext_namespace, rates)
        
        states = []
        for rate_name, rate in self.items():
            namespace.update(rate.namespace)
            states.append(rate.state)
        
        #Save attributes
        self.namespace = tuple(namespace)
        self.states    = tuple(states)
        #Freeze
        self.freeze()

