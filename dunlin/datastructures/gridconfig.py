from numbers import Number
from typing import Union

import dunlin.utils as ut
from .bases               import DataDict, DataValue
from .coordinatecomponent import CoordinateComponentDict

class GridConfig(DataValue):
    def __init__(self,
                 all_names: set, 
                 coordinate_components: CoordinateComponentDict,
                 name: str,
                 config: Union[dict, tuple], 
                 children: list[str]=None,
                 ) -> None:

        #Extract step and spans based on format
        if hasattr(config, 'items'):
            step  = config['step']
            spans = config['spans']
        else:
            try:
                step, *spans = config
            except:
                msg  = 'Could not unpack grid configuration. Expected: '
                msg += 'step, <span0>, <span1>...'
                raise ValueError(msg)
        
            if len(spans) == coordinate_components.ndims:
                spans = dict(zip(coordinate_components.axes, spans))
            else:
                msg  = f'Expected {coordinate_components.ndims} dimensions '
                msg += f'but configuration for "{name}" is for {len(spans)} dimensions.'
                raise ValueError(msg)
                
        #Check step size
        if not isinstance(step, Number):
            msg = f'Expected step size to be a number. Received: {step}'
            raise ValueError(msg)
        elif step <= 0:
            msg = f'Expected step size to be positive. Received: {step}'
        
        #Check no. of spans == ndims
        if set(coordinate_components.keys()) != set(spans):
            msg  = f'Expected spans for {coordinate_components.axes}. '
            msg += f'Received {spans}'
            raise ValueError(msg)
        
        #Check bounds
        for axis, span in spans.items():
            try:
                lb, ub = span
            except:
                msg  = 'Could not unpack span. '
                msg += f'Expected (lower_bound, upper_bound). Received {span}'
                raise ValueError(msg)
                
            if lb >= ub:
                msg  = 'Lower bound more than/equal to upper bound. '
                msg += 'Received {span}'
                raise ValueError(msg)
            
            lb_, ub_ = coordinate_components[axis]
            if lb < lb_:
                msg = f'Lower bound exceeds coordinate {axis} components. '
                msg = f'Expected {lb_}. Received {lb}.'
                raise ValueError(msg)
            if ub > ub_:
                msg = f'Upper bound exceeds coordinate {axis} components. '
                msg = f'Expected {ub_}. Received {ub}.'
                raise ValueError(msg)
        
        #Check children
        children = [] if children is None else children
        
        for child in children:
            if not ut.is_valid_name(child):
                msg = f'Invalid child name: {child}'
                raise ValueError(msg)
        
        #Reformat spans and steps as list
        #Call the parent constructor
        _spans = [list(spans[axis]) for axis in coordinate_components.axes]
        _config = [step, *_spans]
        super().__init__(all_names, name, _config=_config, _children=list(children))
    
    @property
    def ndims(self) -> int:
        return len(self._config) - 1
        
    @property
    def children(self) -> list[str]:
        return self._children
    
    @property
    def config(self) -> dict:
        return self._config
    
    def to_data(self) -> dict:
        dct = {'config': self._config}
        
        if self.children:
            dct['children'] = self.children
        return dct
    
class GridConfigDict(DataDict):
    itype = GridConfig
    
    def __init__(self, 
                 all_names        : set,
                 coordinate_components: CoordinateComponentDict, 
                 mapping              : dict
                 ) -> None:
        
        self.coordinate_components = coordinate_components
        
        self._check(mapping)
        self._data = mapping
    
    def _check(self, dct, is_top_level=True):
        if 'min' not in dct:
            msg = 'Grid config is missing "min".'
            raise ValueError(msg)
        
        if 'max' not in dct:
            msg = 'Grid config is missing "max".'
            raise ValueError(msg)
        
        
        ndims = self.coordinate_components.ndims
        
        if len(dct['min']) != ndims:
            msg = f'Expected "min" to have length {ndims}. Received : {dct["min"]}'
            raise ValueError(msg)
        
        if len(dct['max']) != ndims:
            msg = f'Expected "min" to have length {ndims}. Received : {dct["max"]}'
            raise ValueError(msg)
        
        if is_top_level and 'step' in dct:
            msg = '"step" for grid config can only be provided at the top level.'
        
        for child in dct.get('children', {}).values():
            
            self._check(child, is_top_level=False)
        
            