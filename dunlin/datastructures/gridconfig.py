from numbers import Number
from typing import Union

import dunlin.utils as ut
from .bases               import Tree
from .coordinatecomponent import CoordinateComponentDict

class GridConfig(Tree):
    recurse_at = 'children'
    
    def __init__(self,
                 all_names             : set, 
                 name                  : str,
                 coordinate_components : CoordinateComponentDict,
                 step                  : Number,
                 min                   : list[Number],
                 max                   : list[Number],
                 children              : dict   = None,
                 _parent_step          : Number = None
                 ) -> None:
        
        if not isinstance(step, Number):
            msg = f'Step size for grid config must be a number. Received {type(step)}.'
            raise TypeError(msg)
        if _parent_step is not None:
            if step >= _parent_step:
                msg = f"Step size of child grid {name} is larger than its parent's."
                raise ValueError(msg)
        
        ndims = coordinate_components.ndims
        
        if len(min) != ndims:
            if name is None:
                msg = f'Lower bound {min} must have length {ndims}.'
            else:
                msg = f'Lower bound {min} of child {name} must have length {ndims}.'
            raise ValueError(msg)
        
        if len(max) != ndims:
            if name is None:
                msg = f'Upper bound {max} must have length {ndims}.'
            else:
                msg = f'Upper bound {max} of child {name} must have length {ndims}.'
            raise ValueError(msg)

        for lb, ub in zip(min, max):
            if lb >= ub:
                if name is None:
                    msg = f'Lower bound {lb} >= upper bound {ub} in grid config.'
                else:
                    msg = f'Lower bound {lb} >= upper bound {ub} in child grid config {name}.'
                raise ValueError(msg)

        #Call the parent constructor 
        super().__init__(all_names, 
                         name, 
                         coordinate_components,
                         step         = step, 
                         min          = tuple(min),
                         max          = tuple(max),
                         children     = children,
                         _parent_step = step 
                         )
    
    def __getitem__(self, key: str):
        return self.children[key]
    
    def to_dict(self) -> dict:
        dct = {'step'     : self.step,
               'min'      : list(self.min),
               'max'      : list(self.max),
               }
        if self.children:
            dct['children'] = {k: v.to_dict() for k, v in self.children.items()}
        
        return dct
    
# class GridConfigDict(DataDict):
#     itype = GridConfig
    
#     def __init__(self, 
#                  all_names        : set,
#                  coordinate_components: CoordinateComponentDict, 
#                  mapping              : dict
#                  ) -> None:
        
#         self.coordinate_components = coordinate_components
        
#         self._check(mapping)
#         self._data = mapping
    
#     def _check(self, dct, is_top_level=True):
#         if 'min' not in dct:
#             msg = 'Grid config is missing "min".'
#             raise ValueError(msg)
        
#         if 'max' not in dct:
#             msg = 'Grid config is missing "max".'
#             raise ValueError(msg)
        
        
#         ndims = self.coordinate_components.ndims
        
#         if len(dct['min']) != ndims:
#             msg = f'Expected "min" to have length {ndims}. Received : {dct["min"]}'
#             raise ValueError(msg)
        
#         if len(dct['max']) != ndims:
#             msg = f'Expected "min" to have length {ndims}. Received : {dct["max"]}'
#             raise ValueError(msg)
        
#         if is_top_level and 'step' in dct:
#             msg = '"step" for grid config can only be provided at the top level.'
        
#         for child in dct.get('children', {}).values():
            
#             self._check(child, is_top_level=False)
        
            