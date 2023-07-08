import pandas as pd
from typing import Union

from dunlin.datastructures.bases import Table

class StateDict(Table):
    itype        = 'States' 
    can_be_empty = False
    
    def __init__(self,  
                 all_names : set,
                 mapping   : Union[dict, pd.DataFrame],
                 ) -> None:
        super().__init__(all_names, mapping)
        
class ParameterDict(Table):
    itype        = 'Parameters'
    can_be_empty = False
    
    def __init__(self, 
                 all_names : set,
                 mapping   : Union[dict, pd.DataFrame],
                 ) -> None:
        super().__init__(all_names, mapping)
        

    
    