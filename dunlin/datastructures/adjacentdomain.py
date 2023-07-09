import pandas as pd

import dunlin.utils             as ut
from .bases               import DataValue, DataDict
from .coordinatecomponent import CoordinateComponentDict
from .domain              import DomainDict

class AdjacentDomain(DataValue):
    def __init__(self,
                 all_names             : set,
                 coordinate_components : CoordinateComponentDict,
                 domains               : DomainDict,
                 domain_pairs          : dict,
                 name                  : str,
                 *domain_pair          : list[str, str]                
                 ):
        
        domain_pair_ = frozenset(domain_pair)
        
        if domain_pair_ in domain_pairs:
            msg  = f'Repeated domain pair found: {domain_pair}.'
            msg += f'This pair was found in {name} and {domain_pairs[domain_pair_]}.'
            raise ValueError(msg)
        
        super().__init__(all_names,
                         name,
                         domain_pair     = domain_pair_,
                         domain_pair_ori = domain_pair
                         )
        
        domain_pairs[domain_pair_] = name
    
    def to_dict(self) -> dict:
        dct = {self.name: list(self.domain_pair_ori)}
        
        return dct
    
class AdjacentDomainDict(DataDict):
    itype = AdjacentDomain
    
    def __init__(self,
                 all_names             : set,
                 coordinate_components : CoordinateComponentDict,
                 domains               : DomainDict,
                 mapping               : dict[str, list[str, str]],
                 ):
        
        domain_pairs = {}
        
        super().__init__(all_names, 
                         mapping, 
                         coordinate_components, 
                         domains, 
                         domain_pairs
                         )
        
        self.domain_pairs = domain_pairs
    