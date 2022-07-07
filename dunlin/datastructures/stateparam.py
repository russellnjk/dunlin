from dunlin.utils.typing         import Dflike
from dunlin.datastructures.bases import _BDict

class StateDict(_BDict):
    itype = 'states' 
    
    def __init__(self, mapping: Dflike, 
                 ext_namespace: set
                 ) -> None:
        super().__init__('states', mapping, ext_namespace)
        
class ParameterDict(_BDict):
    itype = 'parameters'
    
    def __init__(self, mapping: Dflike, 
                 ext_namespace: set
                 ) -> None:
        super().__init__('parameters', mapping, ext_namespace)
        

    
    