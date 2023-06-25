import numpy as np
from collections import namedtuple

###############################################################################
#Dunlin Exceptions
###############################################################################
class TriggerError(Exception):
    def __init__(self, trigger_type):
        super().__init__(f'Invalid trigger_type: {trigger_type}')

###############################################################################
#Main Instantiation Algorithm
###############################################################################
evs_tup = namedtuple('rhsevent', 'ref name trigger_func assign_func delay persistent priority')

def make_events(rhsevents: dict[str, evs_tup]):
    events = [make_event(ev_data) for ev_data in rhsevents.values()]
    return events
    
def make_event(ev_data: evs_tup):
    ev_obj = Event(name         = ev_data.name, 
                   trigger_func = ev_data.trigger_func, 
                   execute      = ev_data.assign_func, 
                   delay        = ev_data.delay,
                   persistent   = ev_data.persistent,
                   priority     = ev_data.priority,
                   ref          = ev_data.ref
                   )

    ev_obj._data = ev_data
    return ev_obj
    
###############################################################################
#Classes
###############################################################################
class Event():
    cache = []
    
    def __init__(self, name='', trigger_func=None, execute=None, delay=0, persistent=True, priority=0, ref=None, _parent=None):
        #For scipy
        self.terminal  = True
        self.direction = 1
        
        #For printing
        self.name   = name
        self.ref    = ref
        
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
        return f'{type(self).__name__}<{self.ref}: {self.name}>'
    
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