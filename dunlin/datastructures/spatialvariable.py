import re
from typing import Union

import dunlin.utils as ut
from dunlin.datastructures.bases       import NamespaceDict
from dunlin.datastructures.variable    import Variable
from dunlin.datastructures.compartment import CompartmentDict

class SpatialVariable(Variable):
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, 
                 ext_namespace : set, 
                 compartments  : CompartmentDict,
                 name          : str, 
                 expr          : Union[str, int, float]
                 ) -> None:
        
        #Call the parent constructor and unfreeze
        super().__init__(ext_namespace, name, expr)
        self.unfreeze()
        
        states = compartments.state2compartment.keys()
        
        #Reformat the expr
        pattern = '[a-zA-Z]\w*'
        states  = self.reactants + self.products
        repl    = lambda match: '{' + match[0] + '}' if match[0] in names else match[0]
        
        
        #Refreeze
        self.freeze()
        

class SpatialVariableDict(NamespaceDict):
    itype = Variable
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, 
                 ext_namespace : set, 
                 compartments  : CompartmentDict,
                 variables     : dict
                 ) -> None:
        
        namespace = set()

        #Make the dict
        super().__init__(ext_namespace, variables)
        
        for variable_name, variable in self.items():
            namespace.update(variable.namespace)
        
        #Save attributes
        self.namespace = tuple(namespace)
        
        #Freeze
        self.freeze()
    
