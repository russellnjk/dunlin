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
curvefitters = dn.fit_model(model, 
                            data, 
                            runs = 3, 
                            algo = 'simulated_annealing',
                            lazy = True
                            )

#This is required if lazy = True
curvefitter_lst = []

for i, cf in enumerate(curvefitters):
    fig, AX = dn.figure(2, 2)
    
    fig.suptitle(f'Run {i}')
    
    AX[0].set_title('x')
    AX[1].set_title('S')
    AX[2].set_title('H')
    AX[3].set_title('S vs x')
    
    cf.plot_result(AX[0], 'x')
    cf.plot_result(AX[1], 'S')
    cf.plot_result(AX[2], 'H')
    cf.plot_result(AX[3], ('S', 'x'))
    
    curvefitter_lst.append(cf)

sorted_curvefitters = sorted(curvefitter_lst, key=lambda cf: cf.best_objective)
print(sorted_curvefitters)

#To manually instantiate a curvefitter
#cf = dn.Curvefitter(model, data)
