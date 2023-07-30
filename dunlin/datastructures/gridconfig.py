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
    
    def to_dict(self) -> dict:
        dct = {'step'     : self.step,
               'min'      : list(self.min),
               'max'      : list(self.max),
               }
        if self.children:
            dct['children'] = {k: v.to_dict() for k, v in self.children.items()}
        
        return dct
