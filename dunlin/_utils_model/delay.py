import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from numba import njit
from numba.typed import List, Dict
from numba.experimental import jitclass
from numba import int32, float64

plt.close('all')

spec = [('len_y', int32),
        ('n', int32),
        ('t_hist', float64[:]),       
        ('y_hist', float64[:, :]), 
        ('static', float64[:])
        ]

@jitclass(spec)
class DDE():
    def __init__(self, static, len_y=1):
        # self.t_hist = 
        self.len_y  = len_y
        self.static = static
        self.reset_hist()
        
    def reset_hist(self):
        self.t_hist, self.y_hist = self.template_hist()
        self.n                   = 0
    
    def extend_hist(self):
        t_hist_, y_hist_ = self.template_hist()
        
        self.t_hist = np.concatenate((self.t_hist, t_hist_))
        self.y_hist = np.concatenate((self.y_hist, y_hist_), axis=1)
    
    def template_hist(self):
        t_hist = np.ones(1000)*np.inf
        y_hist = np.ones((1000, self.len_y))
        
        return t_hist, y_hist
    
    def update_hist(self, t, y):
        n = self.n
        if n > len(self.t_hist):
            self.extend_hist()
        
        self.t_hist[n]   = t
        self.y_hist[:,n] = y
        
    def __call__(self, t, y, delay_0=10):
        self.update_hist(t, y)
        
        return self.static(t, y, self.t_hist, self.y_hist, delay_0)
    
@njit
def static(t, y, t_hist, y_hist, delay_0=10):
    x0 = y[0]
    
    if t < delay_0:
        y_delay   = np.array([[1.0], [1.0]])
        
    else:
        y_delay  = get_y_at_t(np.array([t-delay_0, t-20]), t_hist, y_hist)
    
    #Unpack the delayed y
    y_delay_0 = y_delay[0]
    y_delay_1 = y_delay[1]
    
    x0_delay_0 = y_delay_0[0]
    x0_delay_1 = y_delay_0[0]

    dx0 = -0.1*x0_delay_0 + 10
    
    return np.array([dx0])
    
def f_hist_0(t, y):
    return np.array([1])

def get_y_at_t(t, t_hist, y_hist):

    t_arr = np.array(t_hist)
    y_arr = np.array(y_hist).T
    
    interpolated = _interpolate(t, t_arr, y_arr)
    
    return interpolated

@njit
def _interpolate(t, t_arr, y_arr):
    interpolated = np.zeros((len(t), len(y_arr)))
    
    for i in range(len(y_arr)):
        interpolated[:,i] = np.interp(t, t_arr, y_arr[i])
    
    return interpolated

def f1(t, y, hist, delay_0=10):
    x0 = y[0]
    
    if t < delay_0:
        y_delay   = np.array([[1.0], [1.0]])
        
    else:
        y_delay  = get_y_at_t(np.array([t-delay_0, t-20]), hist)
    
    #Unpack the delayed y
    y_delay_0 = y_delay[0]
    y_delay_1 = y_delay[1]

    hist[t] = y
    
    x0_delay_0 = y_delay_0[0]
    x0_delay_1 = y_delay_0[0]

    dx0 = -0.1*x0_delay_0 + 10
    
    return np.array([dx0])

def f0(t, y):
    x0 = y[0]


    dx0 = -0.1*x0 + 10
    
    return np.array([dx0])

def make_hist():
    hist = Dict()
    hist[-40.0] = np.array([1.0])
    return hist
    # t_hist = [-40]
    # y_hist = List()
    # y_hist.append([1])
    
    # return t_hist, y_hist
    
    
tspan = np.linspace(0, 1000, 501)
y0    = np.array([1])
args  = ([0])


r = solve_ivp(f0, [tspan[0], tspan[-1]], 
              y0, 
              t_eval=tspan
              )

t, y,  = r.t, r.y

fig = plt.figure()
ax  = fig.add_subplot(1, 1, 1)

ax.plot(t, y[0])

model = DDE(static, 1)
delay_0 = 14

# r = solve_ivp(f1, [tspan[0], tspan[-1]], 
#               y0, 
#               t_eval=tspan, 
#               args=(delay_0, )
#               )

# t, y,  = r.t, r.y

# ax.plot(t, y[0])

# hist = make_hist()
# delay_0 = 14

# r = solve_ivp(f1, [tspan[0], tspan[-1]], 
#               y0, 
#               t_eval=tspan, 
#               args=(hist, delay_0)
#               )

# t, y,  = r.t, r.y

# ax.plot(t, y[0])

# hist = make_hist()
# delay_0 = 10
# r = solve_ivp(f1, [tspan[0], tspan[-1]], 
#               y0, 
#               t_eval=tspan, 
#               args=(hist, delay_0)
#               )

# t, y,  = r.t, r.y

# ax.plot(t, y[0])

    
    
