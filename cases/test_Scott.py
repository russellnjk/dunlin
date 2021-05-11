import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import matplotlib.ticker as mtick
from   time              import time
from   scipy.optimize    import differential_evolution

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin               as dn
import dunlin.simulation    as sim
import dunlin.curvefit      as cf
import dunlin.traceanalysis as ta
import dunlin.combinatorial as cmb
import dunlin.wrapSSE       as ws

def custom_SSE(y_data, y_model, t_model, scenario):
    variance = 1e-3
    
    Q = y_model[3, -1]
    R = y_model[4, -1]
    M = y_model[5, -1]
    H = y_model[6, -1]
    
    R_frac_model = R/(Q+R+M+H)
    R_frac_data  = exp_data['R_frac'][scenario] 
    
    SSE = (R_frac_data - R_frac_model)**2 / (2*variance)
    
    return -SSE
    
#Set up data
rp_rrna  = 0.53
rrna_rna = 0.86
exrp_rp  = 1.67#Not using this. Conforming to 7459 aa per ribosome.

rp_rna = rp_rrna*rrna_rna

scenarios = ['cAA+Glu', '2um Cm', '4uM Cm', '8uM Cm', '12um Cm', 'cAA+Gly', 'M63+Glu', 'M63+Gly']
exp_data  = {'mu'    : np.array([1.72e-2, 1.57e-2, 1.32e-2, 7.83e-3, 4.33e-3, 1.38e-2, 1e-2,  9e-3 ]),
             'R_frac': np.array([0.302,   0.379,   0.383,   0.458,   0.481,   0.278,   0.199, 0.195])*rp_rna,
             }

plt.style.use(dn.styles['dark_style_multi'])
plt.close('all')

#Read models
model_filename = 'TestScott_1.ini'
model_data     = dn.read_ini(model_filename)

coarse_1   = model_data['coarse_1']['model']

##############################################################################
#Special State in wrap_SSE module
##############################################################################
#Make dataset
dataset = {('ss_R_frac', i, 'Data'): np.array([val]) for i, val in enumerate(exp_data['R_frac'])} 

#Add variance information
dataset[('ss_R_frac', 'Variance')] = 1e-3

exp_data_        = {'coarse_1' : dataset}
cf_objectives    = {'coarse_1': ['ss_R_frac']}
guesses, cf_args = cf.get_sa_args(model_data, exp_data_, cf_objectives)
opt_result       = cf.simulated_annealing(guess=guesses[0], **cf_args)
accepted         = opt_result['accepted']
best             = accepted.iloc[[np.argmax(opt_result['values'])]]

t_fig        = plt.figure()
AX_          = [t_fig.add_subplot(1, 2, i+1) for i in range(2)]
AX           = {'k_cm': AX_[0]}
t_figs, t_ax = ta.plot_steps(accepted, cf_args['step_size'].keys(), AX=AX)

AX_[1].plot(accepted['k_cm'], opt_result['values'], '+')
AX_[1].set_title('posterior vs k_cm')
sim.fs(t_fig)

# ##############################################################################
# #Alternative: Custom SSE in wrap_SSE module
# ##############################################################################
# #This requires more backend knowledge and possibly more hard coding
# dataset     = {('R_frac', i, 'Data'): [val] for i, val in enumerate(exp_data['R_frac'])} 
# dataset[('ss_R_frac', 'Variance')] = 1e-3

# param_index = ws.get_param_index({'coarse_1': coarse_1})[1]
# SSE_calc    = ws.wrap_get_SSE_dataset(coarse_1, dataset={}, custom_func=custom_SSE)
# get_SSE     = ws.wrap_get_SSE(param_index, {'coarse_1': SSE_calc})

# #Test once
# test_params = coarse_1.param_vals.values[0]
# test_SSE    = get_SSE(test_params)

# #Optimize
# count            = 0 
# guesses, cf_args = cf.get_sa_args(model_data, get_SSE)
# guess            = list(guesses.values())[0]
# opt_result       = cf.simulated_annealing(guess=guesses[0], **cf_args)    
# accepted         = opt_result['accepted']

# t_fig        = plt.figure()
# AX_          = [t_fig.add_subplot(1, 2, i+1) for i in range(2)]
# AX           = {'k_cm': AX_[0]}
# t_figs, t_ax = ta.plot_steps(accepted, cf_args['step_size'].keys(), AX=AX)

# AX_[1].plot(accepted['k_cm'], opt_result['values'], '+')
# AX_[1].set_title('posterior vs k_cm')
# sim.fs(t_fig)

##############################################################################
#Simulate
##############################################################################
sim_args   = sim.get_sim_args(model_data)
coarse_1s  = list(coarse_1.states) 
coarse_1e  = list(coarse_1.exvs)
coarse_1e.remove('ss_R_frac')
coarse_1a  = ['x', 'S', 'P'] + coarse_1e

plot_index = {'coarse_1': coarse_1a}
colors     = sim.palette_types['color'](len(scenarios), palette='deep')
line_color = {'coarse_1': dict(zip(range(len(scenarios)), colors))}

#Simulate
simulation_results = sim.integrate_models(sim_args)

simulation_results = sim.integrate_models(sim_args)
figs, AX           = sim.plot_simulation_results(plot_index, 
                                                  simulation_results,
                                                  label       = None,
                                                  color       = line_color
                                                  )

#Overlay Scott et al's data
for (m, r, c) in zip(exp_data['mu'], exp_data['R_frac'], colors):
    AX['coarse_1']['R_frac_vs_mu'].plot([m], [r], 'o', markersize=15, color=c)

#Axes formatting
for fig in figs:
    for ax in fig.axes:
        ax.legend()
        ax.ticklabel_format(style='sci', scilimits=(-2, 3))
        ax.xaxis.set_major_locator(mtick.LinearLocator(4))

AX['coarse_1']['r_sat'].set_ylim(0, 1.05)
AX['coarse_1']['R_frac_vs_mu'].set_ylim(0, 0.25)
