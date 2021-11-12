import numpy as np

###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin.simulate                 as sim
import dunlin._utils_optimize.wrap_SSE as ws
import dunlin._utils_optimize.algos    as ag 
import dunlin._utils_plot.plot         as upp
from dunlin._utils_optimize.params import SampledParam, Bounds, DunlinOptimizationError

###############################################################################
#Plotting
###############################################################################
def plot_dataset(dataset, AX, **data_args):
    global colors
    
    plots = {}
    
    for (dtype, scenario, var), data in dataset.items():
        if dtype != 'Data':
            continue
        
        ax = upp.recursive_get(AX, var, scenario) 
                      
        line_args_  = {**getattr(dataset, 'line_args', {}), **data_args}
        line_args_  = {k: upp.recursive_get(v, scenario, var) for k, v in line_args_.items()}
        
        #Process special keywords
        color = line_args_.get('color')
        if type(color) == str:
            line_args_['color'] = colors[color]
        
        plot_type   = line_args_.get('plot_type', 'errorbar')
            
        #Plot
        if plot_type == 'errorbar':
            if line_args_.get('marker', None) and 'linestyle' not in line_args_:
                line_args_['linestyle'] = 'None'
            
            x_vals = dataset[('Time', scenario, var)]
            y_vals = data
            y_err_ = dataset.get(('Yerr', scenario, var))
            x_err_ = dataset.get(('Xerr', scenario, var))
            
            plots.setdefault(var, {})[scenario] = ax.errorbar(x_vals, y_vals, y_err_, x_err_, **line_args_)
        else:
            raise ValueError(f'Unrecognized plot_type {plot_type}')
        
    return plots