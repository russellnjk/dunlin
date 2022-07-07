from typing import Literal, Optional

import dunlin.utils                    as ut
import dunlin.datastructures.exception as exc
import dunlin.standardfile.dunl.writedunl as wd
from dunlin.utils.typing         import Bnd, OStr
from dunlin.datastructures.bases import _CDict, _CItem

class CoordinateComponent(_CItem):
    allowed = ['x', 'y', 'z']
    def __init__(self, coordinate: Literal['x', 'y', 'z'], 
                 bounds: Bnd, unit: str = 'm'
                 ) -> None:
        
        if coordinate not in self.allowed:
            msg = f'Unexpected value {coordinate}. Expected one of {self.allowed}'
            raise ValueError(msg)
        
        #Check bounds
        lb, ub = bounds
        if lb > ub:
            raise ValueError(f'Lower bound is more than upper bound: {bounds}')
        
        bounds = lb, ub
        
        #Check unit
        if type(unit) != str:
            msg  = 'Expected unit argument to be a string. '
            msg += f'Received: {type(unit).__name__}'
            raise TypeError(msg)
        
        self.coordinate = coordinate
        self.bounds     = bounds
        self.unit       = unit
        
        #It is now safe to call the parent constructor
        super().__init__()
        
        #Freeze
        self.freeze()
    
    def to_dunl(self, multiline=False, **kwargs) -> str:  
        return wd.write_dict(self.to_data(), multiline=multiline, **kwargs)


    def to_data(self) -> dict:
        return {'bounds': list(self.bounds), 'unit': self.unit}
    
class CoordinateComponentDict(_CDict):
    itype = CoordinateComponent
    
    def __init__(self, mapping: dict):
        super().__init__(mapping)
        