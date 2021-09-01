import numpy as np
from pathlib import Path

###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin._utils_model.base_error as dbe
import dunlin._utils_model.ode_coder  as odc

###############################################################################
#Dunlin Exceptions
###############################################################################
class DunlinEventError(dbe.DunlinBaseError):
    @classmethod
    def trigger_type(cls, trigger_type):
        return cls.raise_template(f'Invalid trigger_type: {trigger_type}', 1)

###############################################################################
#Main Instantiation Algorithm
###############################################################################
def make_events(func_data, model_data):
    if model_data.get('events'):
        events = [make_event(event_name, func_data, model_data) for event_name in model_data['events']]
    else:
        events = []
    return events
    
def make_event(event_name, func_data, model_data):
    model_key       = model_data['model_key']
    funcs           = func_data['events']
    funcs           = funcs[event_name]
    trigger_func    = funcs['trigger']
    assignment_func = funcs['assignment']
    event_data      = model_data['events'][event_name]
    
    if hasattr(event_data, 'items'):
        obj = Event(name         = event_name, 
                    trigger_func = trigger_func, 
                    execute      = assignment_func, 
                    delay        = event_data.get('delay', 0), 
                    persistent   = event_data.get('persistent', True), 
                    priority     = event_data.get('priority', 0), 
                    model_key    = model_key
                    )
    else:
        obj = Event(event_name, 
                    trigger_func, 
                    assignment_func, 
                    *event_data[2:], 
                    model_key    = model_key
                    )

    obj._dun            = event_data
    return obj
    
###############################################################################
#Classes
###############################################################################
class Event():
    cache = []
    
    def __init__(self, name='', trigger_func=None, execute=None, delay=0, persistent=True, priority=0, model_key=None, _parent=None):
        #For scipy
        self.terminal     = True
        self.direction    = 1
        
        #For printing
        self.name         = name
        self.model_key    = model_key
        
        #For event assignment
        self.trigger_func = trigger_func
        self._execute     = execute
        self.delay        = delay
        self.persistent   = persistent
        self.priority     = priority
        
        #For tracking 
        self.triggered    = False
        self.timer        = None
        self.record       = []
        self._parent      = _parent
        self.cache.append(self)
    
    def __repr__(self):
        return f'{type(self).__name__}<{self.model_key}: {self.name}>'
    
    def __str__(self):
        return self.__repr__()
    
    def execute(self, t, y, p):
        new_y, new_p = self._execute(t, y, p)
        self.record.append([t, new_p])
        
        curr_r = self.trigger_func(t, y, p)
        new_r  = self.trigger_func(t, new_y, new_p)
        
        #Undo trigger "flip" if execution causes r to fall further below 0
        if new_r < curr_r:
            self.triggered = not self.triggered
            self.direction = -self.direction
        
        return new_y, new_p
    
    def reset(self):
        self.direction    = 1
        self.triggered    = False
        self.timer        = None
        self.record       = []
        
    def setup(self, t, y, p, events_):
        self.reset()
        
        r = self.trigger_func(t, y, p)
        
        if r >= 0:
            new_y, new_p, events_ = self.trigger(t, y, p, events_)
        else:
            new_y, new_p = y, p
        
        return new_y, new_p, events_
    
    def delay_protocol(self, t, events_):
        t_           = t + self.delay
        self.timer   = Event(name         = f'{self.name}, {self.delay}', 
                             trigger_func = lambda t, *args: t - t_,
                             execute      = self.execute,
                             _parent      = self
                             )
        self.timer.remove = True
        events_.append(self.timer)
    
    def __call__(self, t, y, *args):
        r = self.trigger_func(t, y, *args)
            
        return r
    
    def trigger(self, t, y, p, events_):
        new_y, new_p = y, p
        
        if self.delay and self.persistent:
            if not self.triggered:
                self.delay_protocol(t, events_)
            
        elif self.delay:
            if not self.triggered:
                self.delay_protocol(t, events_)
            else:
                self.timer.execute   = None
            
        else:
            if not self.triggered:
                if self.execute:
                    new_y, new_p = self.execute(t, y, p)
                if getattr(self, 'remove', False):
                    events_.remove(self)

        self.triggered = not self.triggered
        self.direction = -self.direction
                
        return new_y, new_p, events_