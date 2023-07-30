from numbers import Number
from typing  import Union

from .bases               import DataDict, DataValue
from .coordinatecomponent import CoordinateComponentDict
from .stateparam          import StateDict, ParameterDict

class BoundaryConditions(DataValue):
    '''Boundary conditions for one state. The input takes the form:
        ```
        {<axis> : {"min": [<value>, <condition_type>],
                   "max": [<value>, <condition_type>],
                   },
         ...
         }
        ```
    ''' 
    def __init__(self,
                 all_names             : set,
                 coordinate_components : CoordinateComponentDict,
                 states                : StateDict,
                 parameters            : ParameterDict,
                 state                 : str,
                 **boundary_conditions,
                 ):
        
        #Check the state
        if state not in states:
            msg = f'Found boundary conditions for unexpected state {state}.'
            raise ValueError(msg)
        
        temp = {}
        for axis, boundary_data in boundary_conditions.items():
            if type(boundary_data) == list:
                boundary_data = {'min': boundary_data, 'max': boundary_data}
                
            for bound, condition in boundary_data.items():
                #Extract the value and condition type
                if type(condition) == list or type(condition) == tuple:
                    value, condition_type = condition
                elif type(condition) == dict:
                    value, condition_type = condition['value'], condition['condition_type']
                else:
                    msg = f'Boundary condition must be a list or dict. Received {type(condition)}.'
                    raise TypeError(msg)
                
                #Check the boundary
                if axis not in coordinate_components.axes:
                    msg = f'Unexpected axis in boundary condition for state {state}: {axis}.'
                    msg = f'{msg} The axis must be one of the coordinate components.'
                    raise ValueError(msg)
                
                if bound != 'min' and bound != 'max':
                    msg = f'Unexpected bound {bound} for state {state}.'
                    msg = f'{msg} Bounds should be "min" or "max".'
                    raise ValueError(msg)
                
                #Check condition_type
                allowed = {'Neumann', 'Dirichlet', 'Robin'}
                if condition_type not in allowed:
                    msg = f'Invalid condition type for state {state} at {axis} {bound}.'
                    msg = f'{msg} : {condition_type}.'
                    msg = f'{msg} Expected one of {allowed}.'
                    raise ValueError(msg)
                
                elif condition_type == 'Robin':
                    raise NotImplementedError('Robin condition not implemented yet.')
                
                #Check value
                if not isinstance(value, Number) :
                    msg = f'Unexpected boundary value for state {state} at {axis} {bound}'
                    msg = f'{msg}: {value}.'
                    msg = f'{msg} Expected a number or parameter.'
                    raise TypeError(msg)
                
                #Update temp
                temp.setdefault(axis, {})[bound] = [value, condition_type]
                
        #Call the parent constructor            
        super().__init__(all_names, 
                         None, 
                         state      = state,
                         conditions = temp
                         )
    
    def to_dict(self) -> dict:
        return {self.state: self.conditions}

class BoundaryConditionDict(DataDict):
    itype = BoundaryConditions
    
    def __init__(self,
                 all_names         : set,
                 coordinate_components : CoordinateComponentDict,
                 states                : StateDict,
                 parameters            : ParameterDict, 
                 mapping               : dict
                 ) -> None:
        super().__init__(all_names, mapping, coordinate_components, states, parameters)
    
    def __getitem__(self,
                    key : tuple[str, Union[str, Number], str]
                    ) -> dict:
        state, axis, bound = key
        
        if type(axis) == str:
            pass
        elif isinstance(axis, Number):
            axis = 'xyz'[axis-1]
        else:
            msg = f'Axis must be a string or a number. Received {type(axis)}.'
            raise ValueError(msg)
        
        boundary_condition = self._data[state].conditions[axis][bound]
        if boundary_condition:
            boundary_condition = dict(zip(['value', 'condition_type'], boundary_condition))
        return boundary_condition
    
    def get(self, 
            state : str,
            axis  : Union[str, Number],
            bound : str
            ) -> Union[None, dict]:
        '''
        This function returns None if the corresponding boundary condition 
        cannot be found.
        '''
        
        if type(axis) == str:
            pass
        elif isinstance(axis, Number):
            axis = 'xyz'[axis-1]
        else:
            msg = f'Axis must be a string or a number. Received {type(axis)}.'
            raise ValueError(msg)
            
        boundary_condition = self._data.get(state, None)
        if boundary_condition:
            boundary_condition = boundary_condition.conditions.get(axis, {}).get(bound)
            
            if boundary_condition:
                boundary_condition = dict(zip(['value', 'condition_type'], boundary_condition))
        return boundary_condition