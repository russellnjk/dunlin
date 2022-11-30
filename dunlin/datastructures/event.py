import re

import dunlin.utils                       as ut
import dunlin.datastructures.exception    as exc
from dunlin.datastructures.bases import NamespaceDict, GenericItem

class Event(GenericItem):
    @staticmethod
    def get_trigger_expr(trigger: str):
        if not ut.isstrlike(trigger):
            raise exc.InvalidDefinition('Event trigger', 
                                        expected=str,
                                        received=trigger
                                        )
        pattern = '([^<>=]*)([<>=][=]?)([^<>=]*)'
        temp    = re.findall(pattern, trigger)
        
        if len(temp) != 1:
            raise exc.InvalidDefinition('Event trigger', 
                                        expected='">" or "<" or "==" between the lhs and rhs',
                                        received=trigger
                                        )
            
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
    def __init__(self, ext_namespace: set, name: str, trigger: str, assign: str, 
                 delay: float = 0, persistent: bool = True, priority: int = 0, 
                 ):
        
        #Format trigger
        trigger_expr = self.get_trigger_expr(trigger)
        
        #Format assign
        assign_expr = self.get_assign_expr(assign)
        
        #Check type for name, delay, persistent and priority
        ut.check_valid_name(name)
        
        if not ut.isnum(delay) and not ut.isstrlike(delay):
            raise exc.InvalidDefinition('Event delay', 
                                        expected='float/int',
                                        received=delay
                                        )
        if type(persistent) != bool:
            raise exc.InvalidDefinition('Event persistence', 
                                        expected=bool,
                                        received=persistent
                                        )
        
        if not ut.isint(priority):
            raise exc.InvalidDefinition('Event priority', 
                                        expected='positive integer',
                                        received=priority
                                        )
        elif priority < 0:
            raise exc.InvalidDefinition('Event priority', 
                                        expected='positive integer',
                                        received=priority
                                        )
            
        
        #Check namespace
        trigger_namespace = ut.get_namespace(trigger_expr, allow_reserved=True)
        assign_namespace  = ut.get_namespace(assign_expr,  allow_reserved=True)
        
        undefined = trigger_namespace.difference(ext_namespace)
        if undefined:
            msg = f'Undefined namespace in event {name} trigger : {undefined}'
            raise NameError(msg)
        undefined = assign_namespace.difference(ext_namespace)
        if undefined:
            msg = f'Undefined namespace in event {name} assign : {undefined}'
            raise NameError(msg)
        
        namespace = trigger_namespace | assign_namespace
        
        #It is now safe to call the parent's init
        super().__init__(ext_namespace, name)
        
        #Save attributes
        self.name         = name
        self.trigger_expr = trigger_expr
        self.assign_expr  = assign_expr
        self.trigger      = trigger
        self.assign       = assign
        self.namespace    = tuple(namespace)
        self.delay        = delay
        self.persistent   = persistent
        self.priority     = priority
        
        #Check name and freeze
        self.freeze()
        
    ###########################################################################
    #Export
    ###########################################################################
    def to_data(self) -> dict:
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
            
        return dct

class EventDict(NamespaceDict):
    itype = Event
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, ext_namespace: set, events: dict) -> None:
        #Make the dict
        super().__init__(ext_namespace, events)
        
        #Freeze
        self.freeze()

