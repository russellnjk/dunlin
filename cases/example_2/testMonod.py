import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

import addpath
import dunlin as dn

plt.ion() 
plt.close('all')

#Read files
model_filename = 'testMonod1.dunl'

loaded  = dn.load_file(model_filename)
model   = loaded.parsed['Monod'] 
data    = loaded.parsed['MonodData']

#Run algorithm
curvefitters = dn.fit_model(model, data, runs=1, algo='simulated_annealing')

fig, AX = dn.figure(2, 2)

cf = curvefitters[0]

cf.plot_result(AX[0], 'x', )

# AX = dict(zip(['x', 'S', 'H', ('S', 'x')], AX_))



# dn.plot_curvefit(AX, 
#                  curvefitters=curvefitters, 
#                  dataset=dataset, 
#                  model=model
#                  )

# best = dn.get_best_optimization(curvefitters)
# print(best.parameters)
# print(best.posterior)
