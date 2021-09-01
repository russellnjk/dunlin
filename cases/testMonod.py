import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy             as np
import pandas            as pd

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin            as dn
import dunlin.simulate   as sim
import dunlin.optimize   as opt
import dunlin.dataparser as dtp
    
plt.style.use(dn.styles['dark_style_multi'])
plt.close('all')

#Read files
data_filename    = 'testMonod1.csv'
model_filename   = 'testMonod1.dun'
dun_data, models = dn.read_file(model_filename)
model            = models['Monod'] 
raw_data         = pd.read_csv('TestMonod1.csv', header=[0, 1], index_col=[0])

np.seterr(all='raise')
#Format the data
#If you have data for more than one state, merge the dictionaries: {**d1, **d2}
dataset = dtp.state2dataset(raw_data, 'x')

#For non-multiplicative 
model_filename   = 'testMonod1.dun'
dun_data, models = dn.read_file(model_filename)
model            = models['Monod'] 
opt_results      = opt.fit_model(model, dataset, n=1, algo='differential_evolution')

#Only one run this time 
best_params, best_posterior = opt_results[0].get_best()

fig, AX = dn.figure(1, 3)
AX_     = dict(zip(model.get_state_names(), AX))

#Plot
opt.integrate_and_plot(model, AX_, opt_results=opt_results, dataset=dataset, guess=':')
