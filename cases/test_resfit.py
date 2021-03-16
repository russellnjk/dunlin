import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import time

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin               as dn
import dunlin.simulation    as sim
import dunlin.curvefit      as cf
import dunlin.traceanalysis as ta

def apply_simulated_annealing(guesses, cf_args):
    traces      = {}
    posteriors  = {}
    opt_results = {}
    for key, guess in guesses.items():
        print('Guess', key)
        start = time.time()
        opt_result = cf.simulated_annealing(guess=guess, **cf_args)
        print(time.time()-start)
        
        accepted   = opt_result['accepted'].iloc[::20] 
        posterior  = opt_result['values'][::20]
        
        traces[key]      = accepted
        posteriors[key]  = posterior
        opt_results[key] = opt_result
        
    return traces, posteriors, opt_results
        
plt.style.use(dn.styles['dark_style_multi'])
plt.close('all')

#Read data
data_filename      = 'M9OD600.csv'
dataset, scenarios = cf.read_csv(data_filename, 'x', header=[1, 2])
exp_data           = {'Monod': dataset}

#Read models
model_filename = 'Monod.ini'
model_data     = dn.read_ini(model_filename)

for model_key, value in model_data.items():
    model = value['model']
    model.init_vals.index  = scenarios
    model.input_vals.index = pd.MultiIndex.from_tuples(zip(scenarios, model.input_vals.index.get_level_values(1)))

# guesses, cf_args = cf.get_sa_args(model_data, exp_data)
# sim_args         = sim.get_sim_args(model_data)

# #Simulation settings
# Monod   = model_data['Monod']['model']
# Monod_s = list(Monod.states)
# Monod_a = Monod_s[:-1]

# plot_index = {'Monod': Monod_a}
# color      = {'Monod': dict(zip(scenarios, ['blue', 'orange', 'green']))}

# # #Test model
# # figs, AX, psim, gsim = cf.integrate_and_plot(plot_index = plot_index, 
# #                                               sim_args   = sim_args,
# #                                               guesses    = guesses, 
# #                                               exp_data   = exp_data,
# #                                               color      = color,
# #                                               label      = 'scenario'  
# #                                               )

# #Run curve-fitting
# traces, posteriors, opt_results = apply_simulated_annealing(guesses, cf_args)

# first = True
# for key, trace in traces.items():
#     index = np.argmax(posteriors[key])
#     best  = trace.iloc[[index]]
    
#     if first:
#         figs, AX, psim, gsim = cf.integrate_and_plot(plot_index = plot_index, 
#                                                      sim_args   = sim_args, 
#                                                      posterior  = best, 
#                                                      guesses    = guesses, 
#                                                      exp_data   = exp_data,
#                                                      color      = color,
#                                                      label      = 'scenario, estimate'
#                                                      )
#         first = False
#     else:
#         _, __,    psim, gsim = cf.integrate_and_plot(plot_index = plot_index, 
#                                                      sim_args   = sim_args, 
#                                                      posterior  = best, 
#                                                      guesses    = best, 
#                                                      exp_data   = exp_data,
#                                                      color      = color,
#                                                      label      = 'scenario, estimate',
#                                                      AX         = AX
#                                                      )
        

# #Post procesing
# for fig in figs:
#     for ax in fig.axes:
#         ax.legend()
#         ax.ticklabel_format(style='sci', scilimits=(-2, 3))
    
# #Trace analysis
# skip           = set([x for key in model_data for x in model_data[key]['model'].params if x not in model_data[key]['step_size']])
# ta_figs, ta_AX = ta.plot_steps(traces, skip=skip)
        