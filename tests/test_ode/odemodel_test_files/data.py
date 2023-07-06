import numpy as np

m0 = {'states'     : {'x0': {'c0': 1},
                      'x1': {'c0': 1},
                      'x2': {'c0': 1},
                      },
      'parameters' : {'p0': {'c0': 0.01},
                      'p1': {'c0': 0.01}
                      },
      'reactions'  : {'r0': ['x0 -> x1', 'p0*x0'],
                      'r1': ['x1 -> x2', 'p1*x1']
                      },
      'events'     : {'e0': ['time==0', ['x0 = 3']],
                      'e6': ['time==800', ['x0 = 3'], 0, True, 1],
                      },
      'tspans'     : {'c0': np.linspace(0, 1000, 101)}
      }

m1 = {'states'    : {'x0': {'c0': 1}
                     },
      'rates'     : {'x0': '-0.1*x0'},
      'submodels' : {'m0': {'ref': 'M0'
                            },
                     },
      }
m2 = {'states'    : {'x0': {'c0': 1}
                     },
      'submodels' : {'m0': {'ref': 'M0'},
                     }
      }
all_data = {'M0': m0, 'M1': m1, 'M2': m2}