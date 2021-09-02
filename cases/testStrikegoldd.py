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

model_filename    = 'testStrikegoldd1.dun'
dun_data, models  = dn.read_file(model_filename)
model             = models['M1'] 
strike_goldd_args = {'observed': ['x1'], 
                     'unknown' : ['p0', 'p1'],
                     'init'    : {'x0': 1, 'x1': 0, 'x2': 0},
                     'inputs'  : {},
                     'decomp'  : []
                     }

result            = dsg.run_strike_goldd(model, **strike_goldd_args)

model  = models['M2'] 
result = dsg.run_strike_goldd(model)

