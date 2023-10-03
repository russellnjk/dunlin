import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

import addpath
import dunlin as dn

plt.ion()
plt.close('all')

#Read files
model_filename    = 'firstorder.dunl'
raw, instantiated = dn.load_file(model_filename)

#Get the model and integrate numerically
model      = instantiated['firstorder'] 
sim_result = model.integrate()

#Plot the results
fig, AX = plt.subplots(2, 2)

AX[0, 0].set_title('x0')
AX[0, 1].set_title('x1')
AX[1, 0].set_title('x2')
AX[1, 1].set_title('x3')

sim_result.plot_line(AX[0, 0], 'x0')
sim_result.plot_line(AX[0, 1], 'x1')
sim_result.plot_line(AX[1, 0], 'x2')
sim_result.plot_line(AX[1, 1], 'x3')

AX[0, 0].legend()
AX[0, 1].legend()
AX[1, 0].legend()
AX[1, 1].legend()
