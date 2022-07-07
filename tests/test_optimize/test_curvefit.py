import matplotlib.pyplot as     plt
import numpy             as     np
import pandas            as     pd

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin                    as dn  
import dunlin.simulate           as sim
import dunlin.optimize.wrap_SSE  as ws
import dunlin.optimize.optimizer as opt
import dunlin.optimize.curvefit  as cf
import dunlin.utils_plot         as upp

plt.ion()
plt.close('all')

m1 = {'states'     : {'x0' : [0, 100],
                      'x1' : [0, 100]
                      },
      'parameters' : {'p0' : [0.1]*2,
                      'p1' : [0.1]*2,
                      'u0' : [1  ]*2,
                      'u1' : [9  ]*2
                      },
      'reactions' : {'g0' : ['   -> x0', 'u0'   ],
                     'g1' : ['   -> x1', 'u1'   ],
                     'r0' : ['x0 ->   ', 'p0*x0'],
                     'r1' : ['x1 ->   ', 'p1*x1']
                     },
      'variables' : {'v0' : 'x0'
                     },
      'tspan'     : {0: np.linspace(0, 110, 111),
                     1: np.linspace(0, 110, 111),
                     },
      'sim_args'  : {'line_args': {'linestyle': '-',
                                   'color'    : {0: 'cobalt',
                                                 1: 'crimson'
                                                 }
                                   },
                     },
      'optim_args': {'free_parameters' : {'u0': {'bounds': [0, 10], 
                                                 'prior': ['uniform', 0, 10]
                                                 },
                                          'u1': {'bounds': [0, 10], 
                                                 'prior': ['uniform', 0, 10]
                                                 },
                                          },
                     'settings'   : {'disp'   : False,
                                     'popsize': 5
                                     },
                     },
      'data_args' : {'dataset' : {'line_args':{'color': {0 : 'cobalt',
                                                         1 : 'crimson'
                                                         },
                                               'marker'    : 'x'
                                               }
                                  }
                    }
      }

all_data = {'M1': m1}

#Read model
model = dn.ODEModel('M1', **m1)

time        = np.linspace(0, 100, 51)
y_data0     = 50 - 50*np.exp(-0.1*time)
y_data1     = 50 + 50*np.exp(-0.1*time)

cols0 = pd.MultiIndex.from_product([['x0', 'x1'], [0]])
cols1 = pd.MultiIndex.from_product([['x0', 'x1'], [1]])
df0 = pd.DataFrame(np.array([y_data0, y_data0]).T, index=time, columns=cols0)
df1 = pd.DataFrame(np.array([y_data1, y_data1]).T, index=time, columns=cols1)

dataset = df0, df1

#Case 1: Function-based parameter estimation
print('Function-based parameter estimation')
cfts = cf.fit_model(model, dataset, algo='differential_evolution')
cft  = cfts[0]

trace = cft.trace
o     = trace.other
assert all(np.isclose(o.x, [5, 5], rtol=1e-3) )
assert type(trace.data) == pd.DataFrame

fig, AX = upp.figure(1, 2)

trace.plot_steps(AX[0], 'posterior', linestyle='-')
trace.plot_steps(AX[1], 'u0', 'u1', linestyle='-')

AX[0].legend()

fig, AX = upp.figure(1, 2)
sr = cft.simulate()
sr.plot_line(AX[0], 'x0')
sr.plot_line(AX[1], 'x1')

#Test master plotting function
fig, AX = upp.figure(1, 2)
AX_ = {'x0': AX[0], 'x1': AX[1]}
r   = cf.plot_curvefit(AX_, curvefitters=cfts, dataset=dataset, model=model)

AX[0].legend()

#Test without curvefit results
fig, AX = upp.figure(1, 2)
AX_ = {'x0': AX[0], 'x1': AX[1]}
r   = cf.plot_curvefit(AX_, dataset=dataset, model=model)

AX[0].legend()

#Test without guess
fig, AX = upp.figure(1, 2)
AX_ = {'x0': AX[0], 'x1': AX[1]}
r   = cf.plot_curvefit(AX_, dataset=dataset, model=model, plot_guess=False)

AX[0].legend()

#Test multiple plots per curvefitter
fig, AX = upp.figure(1, 2)
AX_ = {'x0': AX[0], 'x1': AX[1]}
r   = cf.plot_curvefit(AX_, curvefitters=cfts, n=[0, 5, -1])

AX[0].legend()
