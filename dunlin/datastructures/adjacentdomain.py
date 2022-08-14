import pandas as pd

import dunlin.utils             as ut
from .bases               import TabularDict
from .coordinatecomponent import CoordinateComponentDict
from .domaintype          import DomainTypeDict

class AdjacentDomainsDict(TabularDict):
    is_numeric = False
    
    def __init__(self, 
                 ext_namespace: set,
                 coordinate_components: CoordinateComponentDict,
                 domain_types: DomainTypeDict,
                 mapping: dict,
                 ) -> None:
        
        seen        = set()
        all_domains = domain_types.domains
        data        = {}
        
        for name, value in mapping.items():
            #Check name
            if not ut.is_valid_name(name):
                msg = f'Invalid name provided for adjacent domains: {name}'
                raise ValueError(msg)
            elif name in ext_namespace:
                msg = f'Repeat of namespace {name}.'
                raise ValueError(msg)
            
            #Check value
            if not ut.islistlike(value):
                msg  = 'Expected a list-like pair of adjacent domains. '
                msg += f'Received {value}'
                raise ValueError(msg)
            elif len(value) != 2:
                msg = 'Expected 2 domains. Received {value}'
                raise ValueError(msg)
            
            dmn0, dmn1 = value
            
            if dmn0 not in all_domains:
                a   = [i.name for i in all_domains]
                msg = f'Unexpected domain {dmn0}. Expected one of {a}.'
                raise ValueError(msg)
            elif dmn1 not in all_domains:
                a   = [i.name for i in all_domains]
                msg = f'Unexpected domain {dmn0}. Expected one of {a}.'
                raise ValueError(msg)
            
            temp = dmn0, dmn1
            if temp in seen:
                msg = f'Repeat of adjacent domains {temp}.'
                raise ValueError(msg)
            seen.add(temp)
            
            #Update
            data[name] = list(temp)
        
        #Convert to df
        self.name     = 'adjacent_domains'
        self._df      = pd.DataFrame(data)
        self.n_format = None
    
    def to_data(self) -> dict:
        return self.df.to_dict('list')