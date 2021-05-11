import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import matplotlib.ticker as mtick

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin               as dn
import dunlin.simulation    as sim
import dunlin.curvefit      as cf
import dunlin.dataparser    as dp
import dunlin.traceanalysis as ta

#Generating axes
def make_AX(variables, scenarios):
    AX_ = {state: dict.fromkeys(scenarios) for state in variables}
    AX  = {'coarse_1': AX_}
    
    fig1, AX1_ = sim.figure(2, 5, 10)
    fig2, AX2_ = sim.figure(2, 5, 10)
    
    for i, state in enumerate(variables):
        for scenario in scenarios:
            if type(scenario) == str:
                AX['coarse_1'][state][scenario] = AX1_[i]
            else:
                AX['coarse_1'][state][scenario] = AX2_[i]
                
            AX['coarse_1'][state][scenario].set_title(state)
    
    figs = [fig1, fig2] 
    return figs, AX

#Protocols
def plot_fit(coarse_1a, sim_args, guesses, exp_data=None, **kwargs):
    #This function is mostly hard coded
    figs, AX = make_AX(coarse_1a, scenarios)
    
    try:
        #Plot literature results
        plot_Dai(AX['coarse_1']['R_frac_vs_mu'][3], AX['coarse_1']['r_sat_vs_mu' ][3], AX['coarse_1']['r_sat_vs_R_frac'][3])
    except:
        print('Could not completely plot Dai')
        
    #Simulate and plot
    plot_index        = {'coarse_1': coarse_1a
                         }
    _, __, psim, gsim = cf.integrate_and_plot(plot_index = plot_index, 
                                              sim_args   = sim_args,
                                              guesses    = guesses, 
                                              exp_data   = exp_data,
                                              AX         = AX, 
                                              **kwargs
                                              )
    
    cf.scilimit(figs)
    try:
        format_coarse_exv(AX['coarse_1'])
    except:
        pass
    return figs, AX, psim, gsim

def plot_trace(traces, cf_args, step=1, **kde_args):
    t_figs, t_ax = ta.plot_steps(traces, cf_args['step_size'].keys(), step=step)
    k_figs, k_AX = ta.plot_kde(traces, cf_args['step_size'].keys(), step=step, **kde_args)
    
    sim.scilimit(t_figs)
    sim.scilimit(k_figs)
    
    return t_figs, t_ax, k_figs, k_AX

def make_new_cf_args(cf_args, to_pop_lst):
    #Set up to new cf args
    new_cf_args = cf_args.copy()
    
    new_cf_args['step_size']   = {key: value for key, value in cf_args['step_size'].items() if key not in to_pop_lst}
    new_cf_args['bounds']      = {key: value for key, value in cf_args['bounds'   ].items() if key not in to_pop_lst}
    new_cf_args['iterations'] -= 2500*len(to_pop_lst)
    
    return new_cf_args

def save_figs(index, figs, t_figs, k_figs, folder):
    #Save figs
    if t_figs:
        sim.save_figs(t_figs, folder + '/test_{}_trace_{}.png', index)
    if k_figs:
        sim.save_figs(k_figs, folder + '/test_{}_kde_{}.png', index)
    if figs:
        sim.save_figs(figs, folder + '/test_{}_sim_{}.png', index)
        
        to_trunc = [figs[0].axes[i] for i, state in enumerate(coarse_1a) if 'vs' not in state]
        
        sim.truncate_time(to_trunc, 0, 600)
        sim.save_figs(figs, folder + '/test_{}_trunc_{}.png', index)
    
def save_best(index, best, collected, folder):
    filename = folder + '/best.csv'
    
    try:
        df = pd.read_csv(filename, index_col=[0, 1])
        df = df.append( pd.concat({index: best}, names=['Test']) )
    except:
        df = pd.concat({index: best}, names=['Test'])
    
    df.to_csv(filename)
    collected[index] = best
    
#Plotting   
def dai(mu):
    gradient  = 5.78656638987421
    intercept = 0.03648482880435973
    
    y = gradient*mu + intercept
    
    return y
    
def plot_Dai(ax_R_frac_vs_mu, ax_r_sat_vs_mu, ax_r_sat_vs_R_frac):
    #R_frac from literature
    gradient  = 5.78656638987421
    intercept = 0.03648482880435973
    
    x = np.linspace(0, 0.02, 5)
    y = dai(x)
    
    ax_R_frac_vs_mu.plot(x, y, color='white')
    
    #Saturation rate from literature
    mu    = np.array([0.03,    0.0213,  0.0187,  0.0163,  0.0125,  0.0115,  0.0115, 
                      9.17e-3, 8.33e-3, 7.67e-3, 6.83e-3, 6.33e-3, 5.67e-3, 5.50e-3, 
                      4.83e-3, 3.83e-3, 3.35e-3, 2.17e-3, 5.83e-4
                      ])
    r_sat = np.array([0.958, 0.9,   0.927, 0.865, 0.902, 0.849, 0.888, 
                      0.879, 0.860, 0.879, 0.756, 0.790, 0.751, 0.756, 
                      0.683, 0.590, 0.554, 0.441, 0.168
                      ])
    
    ax_r_sat_vs_mu.plot(mu, r_sat, 'o', color='white')
    
    #For r_sat vs R_frac
    x = gradient*mu + intercept
    ax_r_sat_vs_R_frac.plot(x, r_sat, 'o', color='white')
    
#Post procesing
def format_coarse_exv(model_AX, skip=()):
    to_format = {'R_frac_vs_mu'   : [[0, 2.05e-2], [0, 0.4 ]],
                 'r_sat_vs_mu'    : [[0, 2.05e-2], [0, 1.05]],
                 'r_sat_vs_R_frac': [[0, 0.4    ], [0, 1.05]]
                 }
    
    def helper(ax, xlim, ylim):
        if type(ax) == dict:
            for key, value in ax.items():
                if key in skip:
                    continue
                else:
                    helper(value, xlim, ylim)
        elif type(ax) in [list, tuple]:
            for a in ax:
                a.set_xlim(*xlim)
                a.set_ylim(*ylim)    
        else:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            
    for exv, (xlim, ylim) in to_format.items():
        ax = model_AX[exv]
        
        helper(ax, xlim, ylim)
    
plt.style.use(dn.styles['dark_style_multi'])
plt.close('all')

#Read data
data_filename       = 'data_MCr.csv'
dataset, scenarios_ = dp.read_dataset(data_filename, header=[0, 1, 2])
exp_data            = {'coarse_1': dataset}

#Synthetic exv data
for i in range(4):
    key                       = ('R_frac_ss', len(scenarios_) +i, 'Time' )
    exp_data['coarse_1'][key] = np.array([0, 1e3]) 
exp_data['coarse_1'][('R_frac_ss', 'Variance')] = 1e-3

#Read models
model_filename = 'TestMCr_1.ini'
model_data     = dn.read_ini(model_filename)
print(f'Running test_MCr_3 with {model_filename}')
guesses, cf_args = cf.get_sa_args(model_data, exp_data)
sim_args         = sim.get_sim_args(model_data)

#Simulation settings
coarse_1   = model_data['coarse_1']['model']
coarse_1s  = list(coarse_1.states)
coarse_1e  = list(coarse_1.exvs)
coarse_1a  = ['x', 'S'] + coarse_1e
coarse_1a.remove('R_frac_ss')

#Renumber index
scenarios                = scenarios_ + list(range(len(scenarios_), len(coarse_1.init_vals)))
coarse_1.init_vals.index = scenarios
coarse_1.input_vals.index.set_levels(scenarios, level=0, inplace=True)

#Set colors
palette_1 = [sim.colors['cobalt'], sim.colors['coral'],
             sim.colors['cobalt'], sim.colors['coral'], 
             sim.colors['ocean'],  sim.colors['gold'  ],
             ]
color  = {'coarse_1': dict(zip(scenarios, palette_1)),
          }

vs_marker    = {v: '^' if 'vs' in v else None for v in coarse_1a}
vs_marker    = {estimate: vs_marker for estimate in guesses}
guess_marker = {'coarse_1': {s: vs_marker for s in scenarios}}

# ###############################################################################
# #Test models
# ###############################################################################
# _, __, psim, gsim = plot_fit(coarse_1a, sim_args, guesses, exp_data, color=color, label='scenario', guess_marker=guess_marker)

# # ###############################################################################
# # #Run Curve-fitting
# # ###############################################################################
# #New Test
# collected3 = {}
# folder     = 'testMCr_output'

# #Run curve-fitting
# cf_args['raise_error'] = False

# traces, posteriors, opt_results, best = cf.apply_simulated_annealing(guesses, **cf_args)
    
# #Plot
# plt.close('all')
# figs, AX, psim, gsim       = plot_fit(coarse_1a, sim_args, guesses, exp_data, posterior=best, color=color, label='scenario', guess_marker=guess_marker)
# t_figs, t_ax, k_figs, k_AX = plot_trace(traces, cf_args, bw_adjust=1)

# #Save
# save_figs('scale1n2', figs, t_figs, k_figs, folder=folder)
# save_best('scale1n2', best, collected3, folder=folder)

#New Test
collected2 = {}
folder     = 'testMCr_output'

for guess in guesses.values():
    guess['n_fr'] = 1

#Run curve-fitting
cf_args['raise_error'] = False

traces, posteriors, opt_results, best = cf.apply_simulated_annealing(guesses, **cf_args)
    
#Plot
plt.close('all')
figs, AX, psim, gsim       = plot_fit(coarse_1a, sim_args, guesses, exp_data, posterior=best, color=color, label='scenario', guess_marker=guess_marker)
t_figs, t_ax, k_figs, k_AX = plot_trace(traces, cf_args, bw_adjust=1)

#Save
save_figs('scale1n1', figs, t_figs, k_figs, folder=folder)
save_best('scale1n1', best, collected2, folder=folder)
