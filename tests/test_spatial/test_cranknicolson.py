import matplotlib.pyplot as plt
import numpy             as np
from scipy.integrate       import OdeSolver, DenseOutput
from scipy.optimize        import newton, minimize
from scipy.ndimage import uniform_filter

plt.close('all')
plt.ion()

def rhs(t, y, p):
    dy = -p * y

    return dy

def make_cn(rhs, t0, y0, p0, t1):
    
    def wrapped(y1):
        dt  = t1 - t0 
        dy0 = rhs(t0, y0, p0)
        dy1 = rhs(t1, y1, p0)
        
        residual = y1 - y0 - dt*0.5*(dy0 + dy1) 
        
        return np.sum(residual**2)
    
    y1 = minimize(wrapped, y0).x
    
    return y1

def make_backward(rhs, t0, y0, p0, t1):
    
    def wrapped(y1):
        dt  = t1 - t0 
        dy1 = rhs(t1, y1, p0)
        
        residual = y1 - y0 - dt*dy1
        
        return np.sum(residual**2)
    
    y1 = minimize(wrapped, y0).x
    
    return y1

def make_forward(rhs, t0, y0, p0, t1):
    dt  = t1 - t0 
    dy0 = rhs(t0, y0, p0)
    y1  = y0 + dt * dy0
    
    return y1

def moving_average(array, window=3):
    new_array       = np.zeros(array.shape)
    new_array[:, 0] = array[:, 0]
    w               = np.int32(3/2)
    
    for ii in range(len(array)):
        row = array[ii]
        
        for i in range(1, len(row)):
            start = np.maximum(0, i-w)
            stop  = i + w
            
            new_array[ii, i] = np.mean(row[start:stop])
            
    return new_array

def event0(t, y):
    return y[0] - 0.5

events = [event0]

def solve(get_y1, rhs, tspan, y0, p0, dt, events=events):
    print('Solving...')
    #Internal
    t_start, t_stop = tspan
    t0              = t_start
    y_int           = [y0.copy()]
    p_int           = [p0.copy()]
    t_int           = [t0]
    
    sign0 = [np.sign(event(t0, y0)) for event in events]
    
    while t0 < t_stop:
        #Calculate raw results
        t1 = t0 + dt
        y1 = get_y1(rhs, t0, y0, p0, t1)
        
        #Check events
        for i, event in enumerate(events):
            
            #Check for a change in sign
            sign1_i = np.sign(event(t1, y1))
            if sign1_i == 0:
              raise NotImplementedError() 
            elif sign0[i] != sign1_i:
                #Find a time point where the change occurs
                residual0 = events[i](t0, y0)
                # print(residual0)
                t_event, y_event = find_event(event, get_y1, rhs, t0, y0, p0, dt, )
                # sign0[i] = sign1_i
                print(f'Found an event :) {t_event}\n')
                sign0[i] = sign1_i
            else:
                p1 = p0.copy()
                
        #Update
        y0 = y1
        p0 = p1
        t0 = t1
        
        y_int.append(y1)
        p_int.append(p1)
        t_int.append(t1)
        
    y_int = np.array(y_int).T
    p_int = np.array(p_int).T
    t_int = np.array(t_int)
    
    fig, AX = plt.subplots(1, 2)
    
    AX[0].plot(t_int, y_int[0])
    AX[0].plot(t_int, y_int[1])
    
    AX[1].plot(t_int, p_int[0])
    AX[1].plot(t_int, p_int[1])
    
    return y_int, p_int, t_int

def find_event(event, get_y1, rhs, t0, y0, p0, dt, residual0=None, _tol=None, _depth=0):
    print(t0, y0, dt)
    if _depth > 20:
        msg = t0, y0, dt, residual0
        raise ValueError(msg)
    
    residual0 = event(t0, y0) if residual0 is None else residual0
    sign0     = np.sign(residual0) 
    
    if _tol is None:
        _tol = np.abs(residual0) * 0.01
    
    #Calculate raw results
    t1 = t0 + dt/2
    y1 = get_y1(rhs, t0, y0, p0, t1)
    
    #Check event
    residual1 = event(t1, y1)
    sign1     = np.sign(residual1)
    print(_tol, residual1)
    print()
    if sign1 == 0:
        return t1, y1
    
    elif np.abs(residual1) < np.maximum(np.minimum(_tol, 1e-5), 5e-6):
        return t1, y1
    
    elif sign1 != sign0:
        return find_event(event, 
                          get_y1, 
                          rhs, 
                          t0, 
                          y0, 
                          p0, 
                          dt/2, 
                          residual0 = residual0,
                          _tol      = _tol, 
                          _depth    = _depth+1,
                          )
      
    else:
        return find_event(event, 
                          get_y1, 
                          rhs, 
                          t1, 
                          y1, 
                          p0, 
                          dt/2,
                          residual0 = residual1,
                          _tol      = _tol, 
                          _depth    = _depth+1,
                          )
        
#External
tspan = [0, 50]
y0    = np.array([1, 1])
p0    = np.array([0.1, 0.5])
dt    = 0.1

solve(make_forward, rhs, tspan, y0, p0, dt)

solve(make_backward, rhs, tspan, y0, p0, dt)

solve(make_cn, rhs, tspan, y0, p0, dt)

# dt    = 0.01

# solve(make_forward, rhs, tspan, y0, p0, dt)

# solve(make_backward, rhs, tspan, y0, p0, dt)

# solve(make_cn, rhs, tspan, y0, p0, dt)

# t = find_event(event0, 
#                 make_backward, 
#                 rhs, 
#                 t0 = tspan[0], 
#                 y0 = y0, 
#                 p0 = p0, 
#                 dt = 8
#                 )
# print(t)
