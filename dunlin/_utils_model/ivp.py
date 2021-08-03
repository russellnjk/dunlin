import numpy as np
from numba           import njit
from pathlib         import Path
from scipy.integrate import solve_ivp

###############################################################################
#Non-Standard Imports
###############################################################################
try:
    from  .base_error  import DunlinBaseError
    from  .custom_eval import safe_eval as eval
except Exception as e:
    if Path.cwd() == Path(__file__).parent:
        from  base_error  import DunlinBaseError
        from  custom_eval import safe_eval  as eval
    else:
        raise e

###############################################################################
#Dunlin Exceptions
###############################################################################   
class IVPError(DunlinBaseError):
    pass

###############################################################################
#Main Algorithm
###############################################################################   
def integrate(func, tspan, y0, p, 
              events=(), modify=None, scenario=None, 
              overlap=True, include_events=True,
              _sort=True, **kwargs
              ):
    #Preprocessing for time and state variables
    t_eval   = tspan
    t_last   = tspan[0]
    endpoint = tspan[-1]
    t_eval   = tspan
    interval = np.array([tspan[0], endpoint])
    y0_      = y0
    p_       = p.copy()
    t, y     = None, None
    
    #Event and parameter preprocessing
    #Set direction before integration
    events_ = sorted(list(events), key=lambda x: getattr(x, 'priority', 0), reverse=True) if _sort else list(events)
    
    #Run modify if applicable
    if modify:
        y0_, p_ = modify(y0_, p_, scenario)
    

    #Set up events if any
    for index, event in enumerate(events_):
        if hasattr(event, 'setup'):
            y0, p_, events_ = event.setup(t_eval[0], y0_, p_, events_)
    
    #Set up args
    args_   = (p_,)
    
    #Run loop
    while t_last < endpoint:
        r = solve_ivp(func, interval, 
                      y0      = y0_, 
                      t_eval  = t_eval, 
                      events  = events_,
                      args    = args_,
                      **kwargs
                      )
        
        if r.status == -1:
            msg = f'{r.status}\n{r.message}'
            raise IVPError(msg)
        
        tseg, yseg, indices, t_last, y_last = find_events(r, events_, include_events)
        
        t_rem                       = t_eval[len(r.t):]
        t, y, interval, y0_, t_eval = update(t, y, tseg, yseg, t_last, y_last, t_rem, overlap=overlap)
        t_eval = np.concatenate(([t_last], t_rem))
        
        if indices is not None:
            index            = indices[0]
            event            = events_[index]
            y0_, *args_, events_ = event.trigger(t_last, y_last, p_, events_)

    return t, y

###############################################################################
#Supporting Functions for Event Handling
###############################################################################   
def find_events(r, events, include_events=True):
    if r.status:
        #Find which event caused termination
        indices = []
        
        #Only one terminal event can be triggered at a time
        #Can break after finding the first match
        for i, e in enumerate(events):
            tpoints = r.t_events[i]
            
            if e.terminal and len(tpoints):
                indices.append(i)
                break
        
        index  = indices[0]
        y_     = r.y_events[index]
        t_     = r.t_events[index]
        t_last = t_[0]
        y_last = y_[0]
        
        #Get tseg and yseg for updating t and y
        #Get t_last and y_last to set up for the next iteration of integration        
        if include_events and getattr(e, 'triggered', False) == False:
            tseg, yseg     = concat_event(r.t, r.y, t_, y_)
        else:
            tseg, yseg     = r.t, r.y
            
        return tseg, yseg, indices, t_last, y_last 
    else:
        #If no events, then we have the trivial case
        return r.t, r.y, None, r.t[-1], r.y[:,-1]
    
@njit
def concat_event(t, y, t_, y_):
    new_t = np.concatenate((t, t_))
    new_y = np.concatenate((y, y_.T), axis=1)
    
    return new_t, new_y

@njit
def update(t, y, tseg, yseg, t_last, y_last, remaining_t, overlap=True):
    #Use t_last and y_last as the the new initial values
    #Make t_eval by concatenating t_last with remaining_t
    #This part is independent of overlap.
    y0       = y_last
    t_eval   = np.concatenate((np.array([t_last]), remaining_t,))
    interval = np.array([t_eval[0], t_eval[-1]])
    
    #Update the value of t and y
    #This part is dependent on overlap
    if t is None:
        #Assign directly if the arrays have not been initialized
        new_t, new_y = tseg, yseg
    elif t[-1] == tseg[0] and not overlap:
        #Remove the first value of the latest segment to avoid overlaps
        new_t = np.concatenate((t[:-1],   tseg))
        new_y = np.concatenate((y[:,:-1], yseg), axis=1)
    else:
        #Simple concatenation in all other cases
        new_t = np.concatenate((t, tseg))
        new_y = np.concatenate((y, yseg), axis=1)
    
    return new_t, new_y, interval, y0, t_eval

if __name__ == '__main__':
    pass