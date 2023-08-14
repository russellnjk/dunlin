import matplotlib.pyplot as     plt
import numpy             as     np
import pandas            as     pd
import seaborn           as     sns
 
###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin                      as dn 
import dunlin.optimize.optimizer   as opt
import dunlin.optimize.sensitivity as sen     
import dunlin.utils_plot           as upp

from data import all_data

plt.ion()
plt.close('all')



class Mixed(opt.Optimizer, sen.SensitivityMixin):
    pass

def log_likelihood(params):
    #Only the free params will be passed into this function
    return sum([abs(params[0] - 50), abs(params[1]-10)])

nominal      = {'p0': 50, 'p1': 50, 'p2': 50, 'p3': 50, 'p4': 50}
nominal      = pd.DataFrame([nominal])
free_params  = {'p1': {'bounds': [0,   100], 'scale': 'lin',   'prior': ['parameterScaleNormal', 50, 10]},
                'p3': {'bounds': [0.1, 100], 'scale': 'log10', 'prior': ['parameterScaleNormal',  1,  1]}
                }
optr = Mixed(nominal, free_params, log_likelihood)
r = optr.run_sobol(N=1024)

resultdict = r.analysis
# print(resultdict)
resultdict.plot()
r.heatmap()

optr = Mixed(nominal, free_params, log_likelihood)

r = optr.run_dgsm(N=1024)


