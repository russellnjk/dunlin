import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import seaborn           as sns

import addpath
from simulate_test_files.data import all_data

import dunlin as dn
import dunlin.simulate   as sim
import dunlin.utils_plot as upp

plt.ion()
plt.close('all')

#Set up
model  = dn.ODEModel.from_data(all_data, 'M1')
value  = model.states.loc[['c0']].to_dict()
answer = {'m0.x0': {'c0': 1}, 'm0.x1': {'c0': 1}, 'm0.x2': {'c0': 1}, 'x0': {'c0': 1}}
assert value == answer

###############################################################################
#Test Instantiation
###############################################################################
sr = model.simulate()

###############################################################################
#Test Access
###############################################################################
#Use the get method
value = sr.get('m0.x2')
assert value.items()
assert 'm0.x2' in value
assert 'c0'    in value['m0.x2'] 
assert 'm0.x2' in value
assert 'c1'    in value['m0.x2']

#Test for multiple variables and scenarios
value = sr[list(model.state_names)]
assert set(value.keys()) == {'m0.x0', 'm0.x1', 'm0.x2', 'x0'}

#Test indexing
value  = sr['init_m0.x0']
assert 'init_m0.x0' in value
assert 'init_m0.x0' in value

##############################################################################
#Test Plotting
##############################################################################
fig, AX_ = upp.figure(4, 2)

for x, ax in zip(model.state_names, AX_):
    sr.plot_line(ax, x)

bar_args = {'color': {'init_x0'   : 'red', 
                      'init_m0.x0': 'blue',
                      'm0.r0'     : 'purple',
                      'c0'        : 'cobalt',
                      'c1'        : 'crimson'
                      },
            'rot' : -30
            }
sr.plot_bar(AX_[4], ['init_x0', 'init_m0.x0'], **bar_args)
sr.plot_bar(AX_[5], ['init_x0', 'init_m0.x0', 'init_m0.x1'], by='variable', **bar_args)

##############################################################################
#Test MultiIndex Bar Plot
##############################################################################
#Set up
model  = dn.ODEModel.from_data(all_data, 'M3')
sr     = model.simulate()

fig, AX_ = upp.figure(2, 2)

bar_args = {'color': {'init_x0'   : 'red', 
                      'init_x1'   : 'blue',
                      'init_x2'   : 'green',
                      ('c0', 0)   : 'cobalt',
                      ('c1', 0)   : 'crimson',
                      ('c0', 1)   : 'sea',
                      ('c1', 1)   : 'violet',
                      0 : 'olive',
                      1 : 'salmon'
                      },
            'rot' : -30,
            }

sr.plot_bar(AX_[0], ['init_x0', 'init_x1'], by='scenario', **bar_args)
sr.plot_bar(AX_[1], ['init_x0', 'init_x1'], by='variable', **bar_args)

xnames = lambda i: f'{i[0]}+{i[1]}'
ynames = lambda i: 'Inducer : {}'.format(i)
sr.plot_bar(AX_[2], ['init_x0', 'init_x1'], by=1, **bar_args, 
            xnames=xnames, ynames=ynames
            )

xnames = '{}->{}'
ynames = 'Inducer = {}'
width  = lambda ref, scenarios, variables: 0.2
color  = lambda ref, scenarios, variables: {'init_x0'   : 'red', 
                                            'init_x1'   : 'blue',
                                            'init_x2'   : 'green',
                                            ('c0', 0)   : 'cobalt',
                                            ('c1', 0)   : 'crimson',
                                            ('c0', 1)   : 'sea',
                                            ('c1', 1)   : 'violet',
                                            0 : 'olive',
                                            1 : 'salmon'
                                            }
rot    = lambda ref, scenarios, variables: -30
sr.plot_bar(AX_[3], ['init_x0', 'init_x1'], by=1, color=color, rot=rot, 
            xnames=xnames, ynames=ynames, width=width, stacked=True
            )