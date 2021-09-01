import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin               as dn
import dunlin.dataparser    as dtp
import dunlin.optimize      as opt
import dunlin.simulate      as sim
import cell_calculation     as cc


plt.style.use(dn.styles['dark_style_multi'])
plt.close('all')

#Read files
model_filename   = 'MCr/Test2s.dun'
dun_data, models = dn.read_file(model_filename)
raw_data         = pd.read_excel('testMCr.xlsx', index_col=0, header=[0, 1], sheet_name='Simplified')

#Format the data
#If you have data for more than one state, merge the dictionaries: {**d1, **d2}
rdi             = raw_data.index
idx             = [*raw_data.index[raw_data.index < 400][::1]]#, *raw_data.index[raw_data.index > 400][::10]]
raw_data        = raw_data.loc[idx]
raw_data.index += 0

raw_data = dtp.format_multiindex(raw_data)
od       = raw_data['OD600']
mu       = raw_data['mu']
x        = cc.od2g*od
Rfrac    = cc.rfp_od2rfrac(raw_data['RFP'], raw_data['OD600'])
Gfrac    = cc.gfp_od2gfpfrac(raw_data['GFP'], raw_data['OD600'])
dataset  = {**dtp.state2dataset(Rfrac, 'total_R'), 
            **dtp.state2dataset(Gfrac, 'H' ),
            **dtp.state2dataset(x,     'x' ),
            **dtp.state2dataset(mu,    'mu')
            } 

#Preliminary simulation
model     = models['MCr1']
fig1, AX1 = dn.figure(2, 7, len(model.get_state_names()))
fig2, AX2 = dn.figure(2, 6, len(model.exvs))
AX        = AX1 + AX2

to_plot = list(model.get_state_names()) + list(model.exvs)
AX_     = dict(zip(to_plot, AX))

# opt.integrate_and_plot(model, AX_, dataset=dataset)

opt_results = opt.fit_model(model, dataset, n=5, algo='simulated_annealing')
opt.integrate_and_plot(model, AX_, 
                       opt_results=opt_results, 
                       dataset=dataset, 
                       guess=':'
                       )

AX[0].legend(loc='lower right')
for var, ax in AX_.items():
    ax.set_title(var)
    ymax = ax.yaxis.get_data_interval()[1]
    ymax = max(1e-6, ymax)
    ax.set_ylim(0)

t_fig, t_AX = dn.figure(6, 2)
t_AX_       = dict(zip(model.optim_args['free_params'], t_AX))
opt.plot_traces(opt_results, AX=t_AX_)
