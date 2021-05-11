import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import seaborn           as sns
from matplotlib          import get_backend
from scipy.stats         import skewtest, kurtosistest

###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin._utils_plot.axes as uax

###############################################################################
#Import
###############################################################################
def import_trace(files, keys=[], **pd_args):
    traces   = {}
    for i in range(len(files)):
        key         = keys[i+1] if keys else i+1
        traces[key] = pd.read_csv(files[i],  **pd_args)
    return traces

###############################################################################
#Input Type Detection
###############################################################################
def wrap_trace_type(func):
    def helper(traces, *args, **kwargs):
        if type(traces) == dict:
            if type(traces[next(iter(traces))]) == dict:
                return func({1: traces}, *args, **kwargs)
            else:
                return func(traces, *args, **kwargs)
        elif type(traces) == pd.DataFrame:
            return func({1: traces}, *args, **kwargs)
        else:
            raise TypeError(f'Traces must be a Pandas DataFrame or a dict. Detected : {type(traces)}')
    return helper

###############################################################################
#Skewness and Kurtosis
###############################################################################
def check_skewness(traces, ouaxut='df'):
    return scipy_test(skewtest, traces, ouaxut='df')

def check_kurtosis(traces, ouaxut='df'):
    return scipy_test(kurtosistest, traces, ouaxut='df')

def scipy_test(test_func, traces, ouaxut='df'):
    result    = {}
    for label, trace in traces.items():
        test_result = test_func(trace, axis=0, nan_policy='omit')
        
        if ouaxut == 'df':
            variables     = trace.columns.to_list()
            df            = pd.DataFrame( test_result, columns=variables, index=('stats', 'pval'))
            result[label] = df
        else:
            result[label] = test_result
    
    return result

###############################################################################
#Supporting Functions for Wrapping
###############################################################################
def setup_singleplot(traces, skip, figs, AX):
    first_label = next(iter(traces))
    variables   = [variable for variable in traces[first_label].columns.to_list() if variable not in skip]
    n           = len(variables)
    
    figs1 = figs if AX else [plt.figure()]
    if AX:
        AX1   = AX  
    else:
        AX1 = {variables[i]: figs1[0].add_subplot(n//2 + n%2, 2, i+1) for i in range(len(variables))}
    
    return figs1, AX1, variables, first_label

def make_palette(traces, variables, palette, gradient=1, palette_type='light'):
    global palette_types
    if palette:
        if type(palette[next(iter(palette))]) == dict:
            #Assume colors have been fully specified
            palette1 = palette
        else:
            #Assume colors for each trace has been specified
            palette1 = {}
            for label in palette:
                base_color = palette[label]
                colors     = palette_types[palette_type](gradient, base_color)
                palette1[label] = {variable: colors for variable in variables}
                
    else:
        base_colors = palette_types['color'](len(traces), 'muted')
    
        palette1 = {}
        labels   = list(traces.keys())
        for i in range(len(labels)):
            label           = labels[i] if type(labels[i]) != list else tuple(labels[i])
            base_color      = base_colors[i]
            colors          = palette_types[palette_type](gradient, base_color)
            palette1[label] = {variable: colors for variable in variables}
    
    return palette1

###############################################################################
#2-D Plots
###############################################################################
@wrap_trace_type
def plot_steps_2D(traces, variables=None, AX=None, step=1, **line_args):
    figs, AX1 = (None, AX) if AX else make_AX_2D(list(variables))
    check_2D(variables)
    for key, trace in traces.items():
        line_args_ = {**{'label': key}, **line_args}
        _plot('plot', trace, variables, AX1, step=step, **line_args_)
    return figs, AX1

@wrap_trace_type
def plot_kde_2D(traces, variables=None, AX=None, step=1, **line_args):
    figs, AX1 = (None, AX) if AX else make_AX_2D(list(variables))
    check_2D(variables)
    for key, trace in traces.items():
        line_args_ = {**{'label': key}, **line_args}
        _plot('kde', trace, variables, AX1, step=step, **line_args_)
    return figs, AX1    

def check_2D(variables):
    try:
        if not all([len(x) == 2 and type(x) != str and all([type(v) == str for v in x]) for x in variables]):
            raise ValueError('variables argument for 2-D trace plots must contain only pairs of strings.')
    except:
        raise ValueError('variables argument for 2-D trace plots must contain only pairs of strings.')

###############################################################################
#1-D Plots
###############################################################################
@wrap_trace_type
def plot_steps(traces, variables=None, AX=None, step=1, **line_args):
    if type(variables) == dict:
        # variables_list = [ for key, value in variables.items()]
        raise NotImplementedError()
    else:
        figs, AX1 = (None, AX) if AX else make_AX_1D(list(variables))
        check_1D(variables)
        for key, trace in traces.items():
            line_args_ = {**{'label': key}, **line_args}
            _plot('plot', trace, variables, AX1, step=step, **line_args_)
    return figs, AX1

@wrap_trace_type
def plot_hist(traces, variables=None, AX=None, step=1, **line_args):
    figs, AX1 = (None, AX) if AX else make_AX_1D(list(variables))
    check_1D(variables)
    for key, trace in traces.items():
        line_args_ = {**{'label': key}, **line_args}
        _plot('hist', trace, variables, AX1, step=step, **line_args_)
    return figs, AX1

@wrap_trace_type
def plot_kde(traces, variables=None, AX=None, step=1, **line_args):
    figs, AX1 = (None, AX) if AX else make_AX_1D(list(variables))
    check_1D(variables)
    for key, trace in traces.items():
        line_args_ = {**{'label': key}, **line_args}
        _plot('kde', trace, variables, AX1, step=step, **line_args_)
    
    if figs:
        for fig in figs:
            for ax in fig.axes:
                ax.set_xlabel(None)
    return figs, AX1

def check_1D(variables):
    try:
        if not all([type(v) == str for v in variables]):
            raise ValueError('variables argument for 1-D trace plots must contain only strings.')
    except:
        raise ValueError('variables argument for 1-D trace plots must contain only strings.')
        
###############################################################################
#Supporting Functions
###############################################################################
def _plot(plot_type, trace, variables, AX, step=1, **line_args):
    variables_ = variables if variables is not None else variables.columns if type(variables) == pd.DataFrame else variables.keys()
    
    for variable in variables_:
        #Get DataFrames and slice if necessary
        if type(variable) == str:
            args   = [trace[variable].iloc[::step]]
            ax_key = variable
        else:
            args   = [trace[v].iloc[::step] for v in variable]
            ax_key = tuple(variable)
        
        #Get Axes
        try:
            ax = uax.parse_recursive(AX, ax_key, apply=False)
        except:
            raise Exception(f'Could not get Axes object for {variable}')
        
        #Parse line args
        line_args_ = uax.parse_recursive(line_args, variable)
        
        #Plot
        ax_plot(ax, plot_type, *args, **line_args_)
        ax.legend()
    return AX

def ax_plot(ax, plot_type, *args, **kwargs):
    
    if plot_type == 'plot':    
        default = {'marker': '+', 'markersize': 8, 'linestyle': '', 'markeredgewidth': 1.5}
        kwargs_ = {**default, **kwargs}
        ax.plot(*args, **kwargs_)
    elif plot_type == 'hist':
        ax.hist(*args, **kwargs)
    elif plot_type == 'kde':
        sns.kdeplot(*args, ax=ax, **kwargs)
    else:
        raise ValueError('plot_type must be "plot", "hist" or "kde".')
    return

def make_AX_1D(variables):
    if len(variables) <= 4:
        fig, AX = uax.figure(len(variables), 1, len(variables))
        figs    = [fig]
        if len(variables) == 1:
            AX[0].set_title(variables[0])
        else:
            [AX[i].set_ylabel(variables[i]) for i in range(len(variables))]  
    elif len(variables) <= 6:
        n_figs = len(variables)//6 + (len(variables) % 6 > 0)
        figs   = []
        AX     = []
        start  = 0
        for n in range(n_figs):
            n_ax = min(6, len(variables[start: start+6]))
            
            fig, AX_   = uax.figure(3, 2, n_ax)
            AX        += AX_
            start     += 6
            figs.append(fig)
        [AX[i].set_ylabel(variables[i]) for i in range(len(variables))]  
    else:
        n_figs = len(variables)//8 + (len(variables) % 8 > 0)
        figs   = []
        AX     = []
        start  = 0
        for n in range(n_figs):
            n_ax = min(8, len(variables[start: start+8]))
            
            fig, AX_   = uax.figure(4, 2, n_ax)
            AX        += AX_
            start     += 8
            figs.append(fig)
        [AX[i].set_ylabel(variables[i]) for i in range(len(variables))]
    AX = dict(zip(variables, AX))
    
    return figs, AX

def make_AX_2D(variables):
    if len(variables) <= 2:
        fig, AX = uax.figure(len(variables), 1, len(variables))
        figs    = [fig]
        if len(variables) == 1:
            AX[0].set_title(variables[0])
        else:
            [AX[i].set_ylabel(variables[i]) for i in range(len(variables))]
    elif len(variables) < 4:
        fig, AX = uax.figure(2, 2, len(variables))
        figs    = [fig]
        [AX[i].set_ylabel(variables[i]) for i in range(len(variables))]
    else:
        n_figs = round(len(variables)/4 + 0.5)
        figs   = []
        AX     = []
        start  = 0
        for n in range(n_figs):
            n_ax = min(start+4, len(variables[start:start+4]))
            
            fig, AX_   = uax.figure(2, 2, n_ax)
            AX        += AX_
            start     += 4
            figs.append(fig)
        [AX[i].set_ylabel(variables[i]) for i in range(len(variables))]
    AX = dict(zip(variables, AX))
    
    return figs, AX

 