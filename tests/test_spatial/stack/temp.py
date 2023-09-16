import numpy as np

import addpath
import dunlin         as dn 
import dunlin.ode.ivp as ivp
import dunlin.utils   as ut
from dunlin.spatial.stack.masstransferstack import (calculate_advection, 
                                                    calculate_diffusion,
                                                    calculate_neumann_boundary,
                                                    calculate_dirichlet_boundary
                                                    )
from dunlin.datastructures.spatial          import SpatialModelData
from test_spatial_data                      import all_data


spatial_data = SpatialModelData.from_all_data(all_data, 'M0')

scope = {'__array'       : np.array,
         '__zeros'       : np.zeros,
         '__ones'        : np.ones,
         '__concatenate' : np.concatenate,
         '__Neumann'     : calculate_neumann_boundary,
         '__Dirichlet'   : calculate_dirichlet_boundary,
         '__advection'   : calculate_advection,
         '__diffusion'   : calculate_diffusion
         }

with open('Performance test code.txt', 'r') as file:
    code = file.read()

local = {}
exec(code, scope, local)

rhs = local['model_M0']

states     = np.arange(0, 32)
parameters = spatial_data.parameters.df.loc[0].values
tspan      = np.linspace(0, 50, 11)