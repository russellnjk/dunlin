import numpy as np
from scipy.integrate import solve_ivp

###############################################################################
#Main Algorithm
###############################################################################   
def integrate(func, tspan, y0, p0, 
              events=(), modify=None,
              overlap=True, include_events=True,
              **kwargs
              ):

    #Preprocessing for time and state variables
    t_eval   = np.array(tspan, dtype=np.float64)
    t_last   = tspan[0]
    endpoint = tspan[-1]
    interval = np.array([tspan[0], endpoint])
    y0       = np.array(y0, dtype=np.float64)
    p0       = np.array(p0, dtype=np.float64)
    args     = (p0,)
    
    #Check 
    if tspan[0] > 0:
        raise ValueError('tspan starts from more than 0.')
    if len(p0.shape) != 1:
        raise ValueError('parameter p must be 1-dimensional.')
    if len(y0.shape) != 1:
        raise ValueError('parameter p must be 1-dimensional.')
    
    #Event preprocessing
    #Set direction before integration
    events = sorted(events, key=lambda x: getattr(x, 'priority', 0), reverse=True) 
    
    for event in list(events):
        y0, p0, events = event.setup(0, y0, p0, events)
    
    #Create caches
    t_result = []
    y_result = []
    p_result = []
    
    #Run loop
    while t_last < endpoint:
        r = solve_ivp(func, 
                      t_span  = interval, 
                      y0      = y0, 
                      t_eval  = t_eval, 
                      events  = events,
                      args    = args,
                      **kwargs
                      )
        
        if r.status == -1:
            msg = f'{r.status}\n{r.message}'
            raise IVPError(msg)
        elif r.status == 0:
            pseg     = np.tile(p0, (len(r.t), 1)).T
            
            t_result.append(r.t)
            y_result.append(r.y)
            p_result.append(pseg)
            break
        else:
            #Extract the segment and event information
            tseg, yseg, event, t_last, y_last = find_events(r, events, include_events)
            
            #Create/update
            pseg     = np.tile(p0, (len(tseg), 1)).T
            t_rem    = t_eval[len(r.t):]
            t_eval   = np.concatenate((np.array([t_last]), t_rem,))
            interval = t_eval[0], t_eval[-1] 
            
            t_result.append(tseg)
            y_result.append(yseg)
            p_result.append(pseg)
            
            #Trigger the event
            y0, p0, events = event.trigger(t_last, y_last, p0, events)
    
    t = np.concatenate(t_result)
    y = np.concatenate(y_result, axis=1)
    p = np.concatenate(p_result, axis=1)
    
    return t, y, p

###############################################################################
#Supporting Functions for Event Handling
###############################################################################
def find_events(r, events, include_events=True):
    if r.status:
        #Assumes one and only one terminal event took place
        #Find which event caused termination
        for t_event, y_event, event in zip(r.t_events, r.y_events, events):
            #Break after finding the first match
            if len(t_event):
                break
            
        #Get t_last and y_last to set up for the next iteration of integration        
        t_last = t_event[0]
        y_last = y_event[0]
        
        #Get tseg and yseg for updating t and y
        if include_events and event.triggered == False:
            tseg = np.concatenate((r.t, np.array([t_last]) ))
            yseg = np.concatenate((r.y, y_last[:, None]), axis=1)
            
        else:
            tseg, yseg     = r.t, r.y

    else:
        tseg   = r.t
        yseg   = r.y
        t_last = r.t[-1]
        y_last = r.y[:,-1]
        
    return tseg, yseg, event, t_last, y_last
   
class IVPError(Exception):
    pass
