import matplotlib.pyplot as plt
import numpy             as np
import textwrap          as tw
from matplotlib.gridspec import GridSpec
from matplotlib.patches  import Rectangle, FancyArrowPatch
from seaborn             import xkcd_rgb


import addpath
import dunlin       as dn 
import dunlin.utils as ut
from dunlin.spatial.coder              import Coder 
from dunlin.datastructures.spatial     import SpatialModelData
from spatial_data0                     import all_data

spatial_data = SpatialModelData.from_all_data(all_data, 'M0')

###############################################################################
#Instantiate
###############################################################################
print('Test Instantiation')
coder = Coder(spatial_data, numba=False)         

header = '''
import numpy as np_
from numba       import njit  as _njit
from numba.core  import types as _types
from numba.typed import Dict  as _Dict

'''

#Check code
with open('output_rhs.txt', 'w') as file:
    file.write(coder.rhs_code)
    file.write(coder.rhs_dict_code)
    
###############################################################################
#Check Callables
###############################################################################
rhs = coder.rhs

states     = np.ones(124).astype(np.float64)
parameters = np.ones(9).astype(np.float64)
time       = np.float64(0)

print('Call rhs')
dx = rhs(time, states, parameters)

coder1 = Coder(spatial_data)   
dx = coder1.rhs(time, states, parameters)
  
