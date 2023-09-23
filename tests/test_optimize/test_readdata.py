import matplotlib.pyplot as     plt
import numpy             as     np
import pandas            as pd

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin                   as dn
import dunlin.optimize.readdata as rd

plt.close('all')
plt.ion()

###############################################################################
#Test Reading from Dictionary
###############################################################################
config0 = {'file0': {'filename'       : 'test_readdata0.xlsx',
                     'sheet_name'     : 'Sheet1',
                     'index_col'      : 0,
                     'header'         : [0, 1, 2],
                     'scenario_level' : 'scenario',
                     'state_level'    : 'state',
                     'trial_level'    : 'trial'
                     },
           }

config1 = {'file0': {'filename'       : 'test_readdata0.xlsx',
                     'sheet_name'     : 'Sheet1',
                     'index_col'      : 0,
                     'header'         : [0, 1, 2],
                     },
           }

config2 = {'file0': {'filename'       : 'test_readdata0.xlsx',
                     'sheet_name'     : 'Sheet1',
                     'index_col'      : 0,
                     'header'         : [0, 1, 2],
                     },
           'dtype': 'time_response'
           }

config3 = {'file0': {'filename'       : 'test_readdata0.xlsx',
                     'sheet_name'     : 'Sheet1',
                     'index_col'      : 0,
                     'header'         : [0, 1, 2],
                     },
           'dtype': 'wrongtype'
           }

config4 = {'file0': {'filename'       : 'test_readdata0.xlsx',
                     'sheet_name'     : 'Sheet1',
                     'index_col'      : 0,
                     'header'         : [0, 1, 2],
                     },
           'file1': {'filename'       : 'test_readdata0.xlsx',
                     'sheet_name'     : 'Sheet2',
                     'index_col'      : 0,
                     'header'         : [0, 1, 2],
                     },
           }
config5 = {'file0': {'filename'       : 'test_readdata0.csv',
                     'index_col'      : 0,
                     'header'         : [0, 1, 2],
                     },
           }

r0 = rd.read_time_response(config0)
r1 = rd.read_time_response(config1)

for k, v in r0.items():
    assert k in r1
    assert all(v == r1[k])
   
r2 = rd.read_time_response(config2)

try:
    r3 = rd.read_time_response(config3)
except TypeError:
    assert True
else:
    assert False
    
r4 = rd.read_time_response(config4)
r5 = rd.read_time_response(config5)


