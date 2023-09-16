import matplotlib.pyplot as plt
import numpy as np

import addpath
import dunlin
from dunlin.spatial.spatialmodel   import SpatialModel, EventStack
from dunlin.datastructures.spatial import SpatialModelData
from stack.test_spatial_data       import all_data

plt.ion()
plt.close('all')

def plot(t, y, AX, label='_nolabel'):
    for i, ax in enumerate(AX):
        ax.plot(t, y[i], label=label)
        top = np.max(y[i])
        top = top*1.2 if top else 1
        top = np.maximum(top, ax.get_ylim()[1])
        bottom = -top*.05 
        ax.set_ylim(bottom=bottom, top=top)
        
        if label != '_nolabel':
            ax.legend()

# spatial_data = SpatialModelData.from_all_data(all_data, 'M0')
model = SpatialModel.from_data(all_data, 'M0')

###############################################################################
#Test Integration
###############################################################################
fig = plt.figure()
AX  = [fig.add_subplot(1, 3, i+1) for i in range(3)]

ir = model.integrate()[0]

t = ir['time']
y = ir['A'], ir['B'], ir['C']

# plot(t, y, AX, 'Case 1')