import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

import addpath
import dunlin as dn

plt.ion()
plt.close('all')

#Read files
model_filename   = 'firstorder.dunl'

loaded = dn.load_file(model_filename)
model  = loaded.parsed['firstorder'] 

fig, AX_ = dn.figure(2, 2)

AX               = dict(zip(model.state_names, AX_))
AX[('x1', 'x2')] = AX_[-1]

sim_result = dn.simulate_model(model)
dn.plot_line(AX, sim_result)

AX_[0].legend()
