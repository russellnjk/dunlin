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
import dunlin.curvefit      as cf
import dunlin.dataparser    as dp
import dunlin.traceanalysis as ta

#Post procesing
def format_axes(figs):
    for fig in figs:
        for ax in fig.axes:
            ax.ticklabel_format(style='sci', scilimits=(-2, 3))
            ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=4))
        
plt.style.use(dn.styles['dark_style_multi'])
plt.close('all')

#Read data
data_filename      = 'M9OD600.csv'
dataset, scenarios = dp.read_timeseries(data_filename, 'x', header=[1, 2])
exp_data           = {'Monod': dataset}

#Read models
model_filename = 'TestMonod_1.ini'
model_data     = dn.read_ini(model_filename)

for model_key, value in model_data.items():
    model = value['model']
    model.init_vals.index  = scenarios
    model.input_vals.index.set_levels(scenarios, level=0, inplace=True)

guesses, cf_args = cf.get_sa_args(model_data, exp_data)
sim_args         = sim.get_sim_args(model_data)

#Simulation settings
Monod   = model_data['Monod']['model']
Monod_s = list(Monod.states)
Monod_a = Monod_s[:-1]

plot_index = {'Monod': Monod_a}
color      = {'Monod': dict(zip(scenarios, ['blue', 'orange', 'green']))}

# #Test model
# figs, AX, psim, gsim = cf.integrate_and_plot(plot_index = plot_index, 
#                                               sim_args   = sim_args,
#                                               guesses    = guesses, 
#                                               exp_data   = exp_data,
#                                               color      = color,
#                                               label      = 'scenario'  
#                                               )

#Run curve-fitting
traces, posteriors, opt_results, best = cf.apply_simulated_annealing(guesses, cf_args)

#Trace analysis
t_figs, t_AX = ta.plot_steps(traces, variables=list(cf_args['step_size']))

figs, AX, psim, gsim = cf.integrate_and_plot(plot_index = plot_index, 
                                             sim_args   = sim_args, 
                                             posterior  = best, 
                                             guesses    = guesses, 
                                             exp_data   = exp_data,
                                             color      = color,
                                             label      = 'scenario'
                                             )

format_axes(t_figs)
format_axes(figs)
