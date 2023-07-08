import re

import dunlin.utils                       as ut
from dunlin.datastructures.bases import DataDict, DataValue

class Event(DataValue):
    @staticmethod
    def get_trigger_expr(name:str, trigger: str):
        if type(trigger) != str:
            msg = f'Error in instantiating event {name}.'
            msg = f'{msg} Invalid event trigger. Expected a string.'
            msg = f'{msg} Received {type(trigger)}.'
            raise ValueError(msg)
            
        pattern = '([^<>=]*)([<>=][=]?)([^<>=]*)'
        temp    = re.findall(pattern, trigger)
        
        if len(temp) != 1:
            msg = f'Error in instantiating event {name}.'
            msg = f'{msg} Invalid event trigger. Expected ">", "<" or "==" between the lhs and rhs.'
            msg = f'{msg} Received {trigger}'
            raise ValueError(msg)
            
        lhs, op, rhs = temp[0]

        if '<' in op:
            return f'{rhs.strip()} - {lhs.strip()}'
        else:
            return f'{lhs.strip()} - {rhs.strip()}'
    
    @staticmethod
    def get_assign_expr(assign: str):
        #Case 1: The assignment is dict-like
        if ut.isdictlike(assign):
            assign_ = [f'{k} = {v}' for k, v in assign.items()]
        #Case 2: The assignment is string-like (Only one assignment)
        elif ut.isstrlike(assign):
            assign_ = [assign]
        #Case 3: The assignment is list-like
        else:
            assign_ = list(assign)
        
        return assign_
        
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, 
                 all_names  : set, 
                 name       : str, 
                 trigger    : str, 
                 assign     : str, 
                 delay      : float = 0, 
                 persistent : bool = True, 
                 priority   : int = 0, 
                 ):
        
        #Format trigger
        trigger_expr = self.get_trigger_expr(name, trigger)
        
        #Format assign
        assign_expr = self.get_assign_expr(assign)
        
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
        trigger_namespace = ut.get_namespace(trigger_expr, allow_reserved=True)
        assign_namespace  = ut.get_namespace(assign_expr,  allow_reserved=True)
        
        undefined = trigger_namespace.difference(all_names)
        if undefined:
            msg = f'Undefined namespace in event {name} trigger : {undefined}'
            raise NameError(msg)
        undefined = assign_namespace.difference(all_names)
        if undefined:
            msg = f'Undefined namespace in event {name} assign : {undefined}'
            raise NameError(msg)
        
        #It is now safe to call the parent's init
        super().__init__(all_names, 
                         name         = name,
                         trigger_expr = trigger_expr,
                         assign_expr  = assign_expr,
                         trigger      = trigger,
                         assign       = assign,
                         assignments  = assign,
                         delay        = delay,
                         persistent   = persistent,
                         priority     = priority,
                         )
        
    ###########################################################################
    #Export
    ###########################################################################
    def to_dict(self) -> dict:
        dct = {'trigger' : self.trigger,
               'assign'  : self.assign,
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
    def __init__(self, all_names: set, events: dict) -> None:
        #Make the dict
        super().__init__(all_names, events)
        