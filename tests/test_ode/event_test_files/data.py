import numpy as np

m1 = {'states': {'x0': {'c0': 1},
                 'x1': {'c0': 1},
                 'x2': {'c0': 1},
                 },
      'parameters': {'p0': {'c0': 0.01},
                     'p1': {'c0': 0.01}
                     },
      'reactions': {'r0': {'stoichiometry' :{'x0' : -1,
                                             'x1' : 1,
                                             },
                           'rate'          :'p0*x0'
                           },
                    'r1': {'stoichiometry' :{'x1' : -1,
                                             'x2' : 1,
                                             },
                           'rate'          :'p1*x1'
                           }
                    },
      'events': {'e0': ['time==0', ['x0 = 3']],
                 'e1': ['x1 > 0.2', ['x0 = 3']],
                 'e2': ['x1 > 0.2', ['x0 = 3'], 400],
                 'e3': ['x1 > 0.2', ['x0 = 3'], 400, False],
                 'e4': ['x1 < 0.2', ['x0 = 3'], 10],
                 'e5': ['x2 > 2.5', ['x0 = 3']],
                 'e6': ['time==800', ['x0 = 3'], 0, True, 1],
                 'e7': ['time==800', ['x0 = 3'], 0, True, 0]
                 },
      'tspans': {'c0': np.linspace(0, 1000, 101)}
      }

all_data = {'M1': m1}