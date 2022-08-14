from numbers import Number
from typing import Union

import dunlin.utils as ut
from .bases               import GenericItem, GenericDict
from .coordinatecomponent import CoordinateComponentDict

class GridConfig(GenericItem):
    def __init__(self,
                 ext_namespace: set, 
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
            
            spans = dict(zip(coordinate_components.axes, spans))
        
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
        super().__init__(ext_namespace, name, _config=_config, _children=list(children))
        
        #Freeze
        self.freeze()
    
    @property
    def children(self) -> list[str]:
        return self._children
    
    def to_data(self) -> dict:
        dct = {'config': self._config}
        
        if self.children:
            dct['children'] = self.children
        return dct
    
class GridConfigDict(GenericDict):
    itype = GridConfig
    
    def __init__(self, 
                 ext_namespace: set,
                 coordinate_components: CoordinateComponentDict, 
                 mapping: dict
                 ) -> None:
        super().__init__(ext_namespace, mapping,  coordinate_components)
        
        keys = self.keys()
        for item in self.values():
            if item.children:
                for child in item.children:
                    if child not in keys:
                        msg = f'{item.name} is missing child grid {child}.'
                        raise ValueError(msg)
            