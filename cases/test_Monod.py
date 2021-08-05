import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy             as np
import pandas            as pd

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin               as dn
import dunlin.simulation    as sim
import dunlin.optimize      as opt
import dunlin.dataparser    as dtp

#Preprocess data
plt.style.use(dn.styles['dark_style_multi'])
plt.close('all')

#Read files
data_filename    = 'TestMonod1.csv'
model_filename   = 'TestMonod1.dun'
dun_data, models = dn.read_file(model_filename)
model            = models['Monod'] 
raw_data         = pd.read_csv('TestMonod1.csv', header=[0, 1], index_col=[0])

#Format the data
#If you have data for more than one state, merge the dictionaries: {**d1, **d2}
dataset = dtp.state2dataset(raw_data, 'x')

#Run curve fitting
opt_results = opt.fit_model(model, dataset, algo='differential_evolution')

#Plot trace
t_fig, t_AX = dn.figure(3, 1)
t_AX_       = dict(zip(['mu_max', 'k_S', 'yield_S'], t_AX))
opt.plot_opt_results(opt_results, t_AX_)

for ax in t_AX:
    dn.scilimit(ax)

#Number of runs = number of estimates in model.params
#In this case we only have one run
best_params, best_posterior = opt_results[0].get_best(10)

#Check the fit
#Make the axes
fig, AX = dn.figure(1, 3)
AX_     = dict(zip(model.get_state_names(), AX))

opt.integrate_and_plot_opt_results(model, opt_results, AX_, dataset)

for state, ax in zip(model.get_state_names(), AX):
    ax.set_title(state)
    ax.legend()

'''
Future work:
    1. Find a way to store or infer data_line_args.
    2. Wrap integration, plotting and data visualization into one function.
'''

