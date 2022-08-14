import pandas as pd
from typing import Union

from dunlin.datastructures.bases import TabularDict

class StateDict(TabularDict):
    itype = 'states' 
    
    def __init__(self,  
                 ext_namespace: set,
                 mapping: Union[dict, pd.DataFrame],
                 ) -> None:
        super().__init__(ext_namespace, 'states', mapping)
        
class ParameterDict(TabularDict):
    itype = 'parameters'
    
    def __init__(self, 
                 ext_namespace: set,
                 mapping: Union[dict, pd.DataFrame],
                 ) -> None:
        super().__init__(ext_namespace, 'parameters', mapping)
        

    
    