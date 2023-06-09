from numbers import Number
from typing  import Literal, Union

import dunlin.utils as ut
from .bases               import GenericItem, GenericDict
from .coordinatecomponent import CoordinateComponentDict
from .domaintype          import DomainTypeDict
from .stateparam          import StateDict

class BoundaryCondition(GenericItem):
    def __init__(self,
                 ext_namespace         : set(),
                 coordinate_components : CoordinateComponentDict,
                 states                : StateDict,
                 name                  : str,
                 state                 : str,
                 condition             : Union[Number, str],
                 condition_type        : Literal['Neumann', 'Dirichlet', 'Robin'],
                 axis                  : str=None,
                 bound                 : Literal['min', 'max', None]=None
                 ) -> None:
        
        #Check the state
        if state not in states.names:
            msg = f'Found boundary conditions for unexpected state {state}.'
            raise ValueError(msg)
        
        #Check condition_type
        allowed = ['Neumann', 'Dirichlet', 'Robin']
        if condition_type not in allowed:
            msg  = f'Invalid condition type in {name}: {condition_type}.'
            msg += f'Expected one of {allowed}.'
            raise ValueError(msg)
        
        #Parse and check the condition
        if condition_type == 'Robin':
            raise NotImplementedError('Robin condition not implemented yet.')
        else:
            if not ut.isnum(condition):
                msg  = f'Expected a number for boundary condition {name}. '
                msg += 'Received condition of type {type(condition).__name__}.'
                raise TypeError(msg)
        
        #Check the axis
        if axis not in coordinate_components.axes and axis is not None:
            msg = f'Unexpected axis for boundary condition {name}: {axis}'
            raise ValueError(msg)
        
        #Check the bound
        if bound not in ['min', 'max', None]:
            a   = '"min", "max" or left unspecified(i.e. None)'
            msg = f'The "bound" argument must be {a}.'
            msg = f'{msg}. Received {bound} in boundary condition {name}.'
            raise ValueError(msg)
            
        #Check state
        if not ut.is_valid_name(state):
            msg = f'Invalid state name {state} in {name}.'
            raise ValueError(msg)
        
        #Call the parent constructor            
        super().__init__(ext_namespace, 
                         name, 
                         state          = state,
                         condition_type = condition_type,
                         condition      = condition,
                         axis           = axis,
                         bound          = bound
                         )
        
        #Freeze
        self.freeze()
    
    def to_data(self) -> list:
        lst = [self.state, 
               self.condition, 
               self.condition_type, 
               ]
        if self.axis is not None:
            lst.append(self.axis)
            
        if self.bound is not None:
            lst.append(self.bound)
            
        return lst
    
class BoundaryConditionDict(GenericDict):
    itype = BoundaryCondition
    
    def __init__(self,
                 ext_namespace: set,
                 coordinate_components: CoordinateComponentDict,
                 states: StateDict,
                 mapping: dict
                 ) -> None:
        super().__init__(ext_namespace, mapping, coordinate_components, states)

        seen   = {}
        states = []
        for bc in self.values():
            state = bc.state
            axis  = bc.axis
            bound = bc.bound
            
            msg = f'{bc.name} overlaps with one or more boundary conditions.'
            
            if axis is None:
                if state in seen:
                    raise ValueError(msg)
                else:
                    axes        = coordinate_components.axes
                    seen[state] = {axis: {'min': bc, 'max': bc} for axis in axes}
            
            elif bound is None:
                seen.setdefault(state, {})
                if axis in seen[state]:
                    raise ValueError(msg)
                else:
                    seen[state][axis] = {'min': bc, 'max': bc}
            else:
                seen.setdefault(state, {})
                seen[state].setdefault(axis, {})
                if bound in seen[state][axis]:
                    raise ValueError(msg)
                else:
                    seen[state][axis][bound] = bc
            
            states.append(state)
            
        #For further processing or access
        self.states = frozenset(states)
        self.cache  = seen
        self._axes  = dict(enumerate(coordinate_components.axes, start=1))
        #Freeze
        self.freeze()
    
    def find(self, state, axis, bnd=None):
        '''
        This function returns None if the state cannot be found. If the state 
        can be found, the corresponding boundary condition is returned.
        '''
        if ut.isnum(axis):
            if bnd is not None:
                msg = 'bnd argument must be None when axis argument is a number.'
                raise ValueError(msg)
            else:
                axis_ = self._axes[abs(axis)]
                bnd   = 'max' if axis > 0 else 'min'
        else:
            axis_ = axis 
        
        if bnd is None:
            bc = self.cache.get(state, {}).get(axis_, {})
        else:
            bc = self.cache.get(state, {}).get(axis_, {}).get(bnd, None)
            
        return bc
        
        
            