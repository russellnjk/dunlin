import numpy  as np
import pandas as pd

###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin.utils      as ut
import dunlin.utils_plot as upp

###############################################################################
#Frontend Functions
###############################################################################
def simulate_model(model, **kwargs):
    return model.simulate(**kwargs)

def plot_line(AX, sim_result, repeat_label=False, **line_args):
    if ut.islistlike(sim_result):
        result = []
        for sr in sim_result:
            temp = plot_line(AX, sr, **line_args) 
            result.append(temp)
            
            if not repeat_label:
                line_args['label'] = '_nolabel'
                
        return result
    
    plot_result = {}
    for var, ax_dct in AX.items():
        
        if not sim_result.has(var):
            continue
        
        plot_result[var] = sim_result.plot_line(ax_dct, 
                                                var, 
                                                **line_args
                                                )
    
    return plot_result

def plot_bar(AX, sim_result, **bar_args):
    plot_result = {}
    for var, ax_dct in AX.items():
        plot_result[var] = sim_result.plot_bar(ax_dct, 
                                               var, 
                                               **bar_args
                                               )
    
    return plot_result

