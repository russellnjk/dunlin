import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy             as np
import pandas            as pd

import textwrap as tw

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin              as dn
import dunlin.strike_goldd as dsg

from   sympy          import symbols
from   sympy.matrices import Matrix, zeros

# m1, p1 = symbols(['m1', 'p1'])
# synm1, degm, synp1, degp = symbols(['synm', 'degm', 'synp', 'degp'])
# g1, g2, g3, g4 = symbols(['g1', 'g2', 'g3', 'g4'])
# u1 = symbols(['u1'])[0]

# dm1 = synm1 - degm*m1 - u1*g1*g2*g3*g4
# dp1 = synp1*m1 - degp*p1 + u1

# x = Matrix([m1, p1])
# h = Matrix([p1])
# p = Matrix([synm1, degm, synp1, degp, g1, g2, g3, g4])
# f = Matrix([dm1, dp1])
# u = {u1: 0}

# ics = {state: 0 for state in x}

# r = {'h': h, 'u': u, 'p': p, 'f': f, 'ics': ics, 'x': x}
# dsg.sga.strike_goldd(**r)


model_filename   = 'testStrikegoldd1.dun'
dun_data, models = dn.read_file(model_filename)
# model            = models['M1'] 
# model.strike_goldd_args = {'observed': ['x1'], 
#                            'unknown' : ['p0', 'p1'],
#                            'init'    : {'x0': 1, 'x1': 0, 'x2': 0},
#                            'inputs'  : {},
#                            'decomp'  : []
#                            }

# symbolic, r = dsg.convert2symbolic(model)

# dsg.sga.strike_goldd(**r)

model            = models['M2'] 
model.strike_goldd_args = {'observed': ['x'], 
                           'unknown' : ['yield_S'],
                           'init'    : {'x': 1, 'S': 1},
                           'inputs'  : {},
                           'decomp'  : []
                           }

symbolic, r = dsg.convert2symbolic(model)

dsg.sga.strike_goldd(**r)



# model            = models['Monod'] 
# model.strike_goldd_args = {'observed': ['x', 'H'], 
#                             'unknown' : ['mu_max', 'v_H_max'],
#                             'init'    : {'x': 0.05, 'S': 1, 'H': 0},
#                             'inputs'  : {},
#                             'decomp'  : []
#                             }

# symbolic, r = dsg.convert2symbolic(model)

# dsg.sga.strike_goldd(**r)

# print(r)
