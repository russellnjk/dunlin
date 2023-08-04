import re
from numbers import Number

import dunlin.utils as ut
from dunlin.datastructures.bases      import DataDict, DataValue
from dunlin.datastructures.stateparam import StateDict, ParameterDict

class Event(DataValue):
    '''Corresponds to SBML events. However, the formatting for the trigger is 
    restricted to the form `<expr> < 0` or `<expr> > 0` where `<expr>` is a string 
    corresponding to the `trigger` argument in the constructor for this class.
    '''
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, 
                 all_names   : set, 
                 states      : StateDict,
                 parameters  : ParameterDict,
                 name        : str, 
                 trigger     : str, 
                 assign      : dict[str, str], 
                 delay       : float = 0, 
                 persistent  : bool  = True, 
                 priority    : int   = 0, 
                 ):
        
        #Check type for name, delay, persistent and priority
        ut.check_valid_name(name)
        
        if not ut.isnum(delay) and not ut.isstrlike(delay):
            msg = f'Error in instantiating event {name}.'
            msg = f'{msg} Invalid event delay. Expected float or int.'
            msg = f'{msg} Received {type(delay)}.'
            raise ValueError(msg)
            
        if type(persistent) != bool:
            msg = f'Error in instantiating event {name}.'
            msg = f'{msg} Invalid event persistence. Expected bool.'
            msg = f'{msg} Received {type(persistent)}.'
            raise ValueError(msg)
        
        if not ut.isint(priority):
            msg = f'Error in instantiating event {name}.'
            msg = f'{msg} Invalid event priority. Expected a positive integer.'
            msg = f'{msg} Received {type(priority)}.'
            raise ValueError(msg)
            
        elif priority < 0:
            msg = f'Error in instantiating event {name}.'
            msg = f'{msg} Invalid event priority. Expected a positive integer.'
            msg = f'{msg} Received {type(priority)}.'
            raise ValueError(msg)
            
        #Check namespace
        trigger_namespace = ut.get_namespace(trigger, allow_reserved=True)
        
        undefined = trigger_namespace.difference(all_names)
        if undefined:
            msg = f'Undefined namespace in event {name} trigger : {undefined}'
            raise NameError(msg)
        
        #Check trigger
        trigger_namespace = ut.get_namespace(trigger, allow_reserved=True)
        
        for trigger_name in trigger_namespace:
            if trigger_name not in states and trigger_name not in parameters and trigger_name != 'time':
                msg  = f'Error in parsing event {name}. '
                msg += 'Trigger can only contain "time", states or parameters. '
                msg += f'Received: {trigger_name}'
                raise ValueError(msg)
        
        #Check and copy assignments
        if type(assign) != dict:
            msg = 'Event assignment must be a dict. Received {type(assign)}.'
            raise TypeError(msg)
            
        assignments_ = {}
        for lhs, rhs in assign.items():
            if lhs not in states and lhs not in parameters and lhs != 'time':
                msg  = f'Error in parsing event {name}. '
                msg += 'Left-hand side of assignment must be "time", a state or parameter. '
                msg += f'Received: {lhs}'
                raise ValueError(msg)
            
            if type(rhs) != str and not isinstance(rhs, Number):
                msg  = f'Error in parsing event {name}. '
                msg += 'Right-hand side must be string or number.'
                raise TypeError(msg)
            elif isinstance(rhs, Number):
                pass
            else:
                rhs_names = ut.get_namespace(rhs)
                for rhs_name in rhs_names:
                    if rhs_name not in states and rhs_name not in parameters and rhs_name != 'time':
                        msg  = f'Error in parsing event {name}. '
                        msg += 'Right-hand side of assignment can only contain numbers, states and parameters. '
                        msg += f'Received: {lhs}'
                        raise ValueError(msg)
            
            assignments_[lhs] = rhs
                    
        #It is now safe to call the parent's init
        super().__init__(all_names, 
                         name         = name,
                         trigger      = trigger,
                         assignments  = assignments_,
                         delay        = delay,
                         persistent   = persistent,
                         priority     = priority,
                         )
        
    ###########################################################################
    #Export
    ###########################################################################
    def to_dict(self) -> dict:
        dct = {'trigger' : self.trigger,
               'assign'  : self.assignments.copy(),
               }
        
        delay      = self.delay
        persistent = self.persistent
        priority   = self.priority
        
        if delay:
            dct['delay'] = delay
        
        if not persistent:
            dct['persistent'] = persistent
            
        if priority:
            dct['priority'] = priority
        
        dct = {self.name: dct}
        return dct

class EventDict(DataDict):
    itype = Event
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, 
                 all_names  : set, 
                 states     : StateDict, 
                 parameters : ParameterDict,
                 events     : dict
                 ):
        #Make the dict
        super().__init__(all_names, events, states, parameters)
        