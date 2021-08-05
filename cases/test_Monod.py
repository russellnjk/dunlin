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

#Preprocess data
def from_df(raw_data, state, new_state=None):
    def str2num(x):
        try:
            return int(x)
        except:
            try:
                return float(x)
            except:
                return x
            
    #Get time and mean of replicates
    time      = np.array(raw_data.index)
    mean      = raw_data.groupby(axis=1, level=0).mean()
    dataset   = {}
    new_state = state if new_state is None else new_state
    for scenario, y_data in mean.items():
        #Create key and values
        scenario = str2num(scenario)
        y_data   = y_data.values
        data_key = ('Data', scenario, new_state)
        time_key = ('Time', scenario, new_state)
        
        #Assign
        dataset[data_key] = y_data
        dataset[time_key] = time
    return dataset

plt.style.use(dn.styles['dark_style_multi'])
plt.close('all')

#Read files
data_filename    = 'TestMonod1.csv'
model_filename   = 'TestMonod1.dun'
dun_data, models = dn.read_file(model_filename)
model            = models['Monod'] 
raw_data         = pd.read_csv('TestMonod1.csv', header=[0, 1], index_col=[0])

#Format the data
dataset = from_df(raw_data, 'OD600', 'x')

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

#Integrate and plot
sim_results = opt.integrate_opt_result(model, opt_results[0])
sim.plot_sim_results(sim_results, AX_, label='scenario')

#Overlay the data
data_line_args = {**model.sim_args['line_args'], **{'marker': 'o', 'linestyle': 'None'}}
opt.plot_dataset(dataset, AX_, **data_line_args)

for state, ax in zip(model.get_state_names(), AX):
    ax.set_title(state)
    ax.legend()

'''
Future work:
    1. Find a way to store or infer data_line_args.
    2. Wrap integration, plotting and data visualization into one function.
'''

