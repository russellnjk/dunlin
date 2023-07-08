import pandas as pd
from numbers import Number
from typing  import Callable, Literal

import dunlin.utils as ut
import dunlin.standardfile.dunl as sfd
from dunlin.datastructures.bases import Table
    
class CoordinateComponentDict(Table):
    def __init__(self, mapping: dict, n_format: Callable=sfd.format_num):
        _data = {}
        
        for axis, bounds in mapping.items():
            lb, ub  = bounds
            
            #Check bounds
            if lb >= ub:
                raise ValueError(f'Lower bound is not less than upper bound: {lb}, {ub}')
            
            _data[axis] = lb, ub
        
        #Sort _data
        _data = {k: _data[k] for k in sorted(_data)}
        
        #Check that valid coordinate combinations are used
        if len(_data) == 1 and list(_data.keys()) != ['x']:
            msg = f'Expected axes: ["x"]. Received: {list(_data.keys())}'
            raise ValueError(msg)
        elif len(_data) == 2 and list(_data.keys()) != ['x', 'y']:
            msg = f'Expected axes: ["x", "y"]. Received: {list(_data.keys())}'
            raise ValueError(msg)
        elif len(_data) == 3 and list(_data.keys()) != ['x', 'y', 'z']:
            msg = f'Expected axes: ["x", "y", "z"]. Received: {list(_data.keys())}'
            raise ValueError(msg)
        
        
        #Override the parent constructor by directly saving the attributes
        self.name     = 'coordinate_components'
        self._df      = pd.DataFrame(_data)
        self.n_format = n_format
        
    @property
    def ndims(self) -> int:
        return len(self._df.columns)
    
    @property
    def axes(self):
        s = list(self._df.columns)
        return s
    
    def to_data(self) -> dict:
        return self._df.to_dict('list')
    
    @property
    def spans(self) -> dict:
        return self.to_data()
    
    def __contains__(self, point):
        for coordinate, coordinate_component in zip(point, self.values()):
            if not ut.isnum(coordinate):
                msg  = 'Error in determining if {point} is in coordinate components.'
                msg += f' Coordinate {coordinate} is not a number.'
                raise ValueError(msg)
            
            lb, ub = coordinate_component
            if coordinate < lb or coordinate > ub:
                return False
        return True
    
    