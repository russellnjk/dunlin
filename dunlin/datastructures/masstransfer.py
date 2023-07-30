import pandas as pd
from numbers import Number
from typing  import Any

import dunlin.utils as ut
from dunlin.datastructures.bases import DataDict, DataValue
from .stateparam                 import ParameterDict, StateDict
from .coordinatecomponent        import CoordinateComponentDict
from .rate                       import RateDict

class MassTransfer(DataValue):
    itype : str
    
    def __init__(self,  
                 all_names             : set,
                 coordinate_components : CoordinateComponentDict,
                 states                : StateDict,
                 parameters            : ParameterDict,
                 state                 : str,
                 *coefficients_tup,
                 **coefficients_dct
                 ):
        #Check the state
        if state not in states:
            msg = f'State {state} is not in model states.'
            raise NameError(msg)
            
        #Parse and check the cofficients
        if coefficients_tup and coefficients_dct:
            msg = 'Parameters for mass transfer must be specified as a list or dict but a combination of both.'
            msg = f'{msg} Received {coefficients_tup} and {coefficients_dct} for {self.itype} for {state}.'
            raise ValueError(msg)
        
        ndims = coordinate_components.ndims
        
        if coefficients_tup:
            if len(coefficients_tup) == 1:
                coefficients = dict(zip('xyz', coefficients_tup*ndims))
            elif len(coefficients_tup) == ndims:
                coefficients = dict(zip('xyz', coefficients_tup))
            else:
                msg  = f'Expected {ndims}-dimensional coefficient for {self.itype} for {state}. '
                msg += f'One of the mass transfer coefficient for {state} has length {len(coefficients_tup)}.'
                raise ValueError(msg)
        
        else:
            if len(coefficients_dct) == ndims:
                coefficients = dict(coefficients_dct)
            else:
                msg  = 'Expected {ndims} coefficient. '
                msg += 'One of the mass transfer coefficient for {state} has length {len(coefficients_dct)}.'
                raise ValueError(msg)
        
        allowed    = set(parameters.names)
        received   = [v for v in coefficients.values() if not ut.isnum(v,include_strings=True)]
        received   = set(received)
        unexpected = received.difference(allowed)
        
        if unexpected:
            msg  = 'Unexpected coefficients detected for state {state}: {unexpected}'
            msg += 'Ensure that coefficients are parameters or numbers. '
            raise ValueError(msg)
        
        
        #Call the parent constructor
        super().__init__(all_names,
                         name         = None,
                         state        = state,
                         coefficients = coefficients
                         )
    
    def __getitem__(self, axis: str) -> str|Number:
        return self.coefficients[axis]
    
    def get(self, axis: str, default: Any=None) -> None|str|Number:
        return self.coefficients.get(axis, default)
    
    def to_dict(self) -> dict:
        lst = [*self.coefficients.values()]
        
        if len(lst) == 1:
            dct = {self.state: lst[0]}
        elif len(set(lst)) == 1:
            dct = {self.state: lst[0]}
        else:
            dct = {self.state: lst}
        
        return dct
        
class MassTransferDict(DataDict):
    itype: type
    
    def __init__(self, 
                 all_names             : set,
                 coordinate_components : CoordinateComponentDict,
                 states                : StateDict,
                 parameters            : ParameterDict,
                 mapping               : dict|list|str|Number,
                 ) -> None:
        
        super().__init__(all_names, 
                         mapping, 
                         coordinate_components, 
                         states, 
                         parameters
                         )
    
    def __getitem__(self, 
                    key: tuple[str, str| Number]
                    ) -> str|Number:
        state, axis = key
        
        if type(axis) == str:
            pass
        elif isinstance(axis, Number):
            axis = 'xyz'[axis-1]
        else:
            msg = f'Axis must be a string or a number. Received {type(axis)}.'
            raise ValueError(msg)
        
        return self._data[state][axis]
        
    def get(self,
            state   : str, 
            axis    : str|Number,
            default : Any                = None
            ) -> None|str|Number:
        
        if type(axis) == str:
            pass
        elif isinstance(axis, Number):
            axis = 'xyz'[axis-1]
        else:
            msg = f'Axis must be a string or a number. Received {type(axis)}.'
            raise ValueError(msg)
        
        return self._data.get(state, {}).get(axis, default)
    
class Advection(MassTransfer):
    itype = 'advection'

class AdvectionDict(MassTransferDict):
    itype = Advection

class Diffusion(MassTransfer):
    itype = 'diffusion'

class DiffusionDict(MassTransferDict):
    itype = Diffusion

    
    