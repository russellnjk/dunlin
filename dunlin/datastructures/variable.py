from typing import Union

import dunlin.utils                       as ut
from dunlin.datastructures.bases import NamespaceDict, GenericItem

class Variable(GenericItem):
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, ext_namespace: set, name: str, expr: Union[str, int, float]):
        #Parse expression and check
        expr_ori  = self.format_primitive(expr)
        expr      = str(expr).strip()
        namespace = ut.get_namespace(expr, allow_reserved=True)
        undefined = namespace.difference(ext_namespace)
        if undefined:
            raise NameError(f'Undefined namespace: {undefined}.')
        
        #It is now safe to call the parent's init
        super().__init__(ext_namespace, name)
        
        #Save attributes
        self.expr      = expr
        self.expr_ori  = expr_ori
        self.namespace = tuple(namespace)
        
        #Freeze
        self.freeze()
    
    ###########################################################################
    #Export
    ###########################################################################
    def to_data(self) -> str:
        return self.expr_ori

class VariableDict(NamespaceDict):
    itype = Variable
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, ext_namespace: set, variables: dict) -> None:
        namespace = set()

        #Make the dict
        super().__init__(ext_namespace, variables)
        
        for variable_name, variable in self.items():
            namespace.update(variable.namespace)
        
        #Save attributes
        self.namespace = tuple(namespace)
        
        #Freeze
        self.freeze()
    


