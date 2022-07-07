import matplotlib.pyplot as plt
import numpy as np

import addpath
import dunlin.utils as ut
import dunlin.comp as cmp
import dunlin.ode.ode_coder as odc
import dunlin.ode.event    as oev 
import dunlin.ode.ivp       as ivp
from odemodel_test_files.data import all_data

import dunlin.ode.odemodel as dom

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

###############################################################################
#Test Instantiation and Attribute
###############################################################################            
model  = dom.ODEModel.from_data(all_data, 'M0')

value  = model.states.to_dict()
answer = {'x0': {'c0': 1}, 'x1': {'c0': 1}, 'x2': {'c0': 1}}
assert value == answer

value  = model.parameters.to_dict()
answer = {'p0': {'c0': 0.01}, 'p1': {'c0': 0.01}}
assert value == answer

###############################################################################
#Test Integration
###############################################################################
fig = plt.figure()
AX  = [fig.add_subplot(1, 3, i+1) for i in range(3)]

ir = model('c0')

t = ir['time']
y = ir['x0'], ir['x1'], ir['x2']

plot(t, y, AX, 'Case 1')

###############################################################################
#Test Hierarchical Model
###############################################################################
model  = dom.ODEModel.from_data(all_data, 'M1')

value  = model.states.to_dict()
answer = {'m0.x0': {'c0': 1}, 'm0.x1': {'c0': 1}, 'm0.x2': {'c0': 1}, 'x0': {'c0': 1}}
assert value == answer

value  = model.parameters.to_dict()
answer = {'m0.p0': {'c0': 0.01}, 'm0.p1': {'c0': 0.01}}
assert value == answer

fig = plt.figure()
AX  = [fig.add_subplot(2, 2, i+1) for i in range(4)]

ir = model('c0')

t = ir['time']
y = ir['m0.x0'], ir['m0.x1'], ir['m0.x2'], ir['x0']

plot(t, y, AX, 'Case 2')
assert ir['init_m0.x0'] == 3

time = np.array([0, 1, 2], dtype=np.float64)
states = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=np.float64)
parameters = np.array([[0.01, 0.01, 0.01], [0.01, 0.01, 0.01]], dtype=np.float64)
ir.dct(time, states, parameters)

t, y, p = ir._args
r = ir.dct(t, y, p)

value  = ir['m0.r0']
answer = np.array([3.00000000e-02, 1.81959198e-02, 1.10363831e-02, 6.69390476e-03,
                   4.06004253e-03, 2.46255731e-03, 1.49361415e-03, 9.05926084e-04,
                   5.49471311e-04, 3.33272190e-04, 2.02139547e-04, 1.22604229e-04,
                   7.43631243e-05, 4.51036643e-05, 2.73567313e-05, 1.65927357e-05,
                   1.00639909e-05, 1.00639909e-05, 3.00000000e-02, 1.81959679e-02,
                   1.10364249e-02, 6.69394669e-03, 4.06009157e-03])
assert np.allclose(answer, value, atol=1e-4)

# print(ir.eval_dct)
