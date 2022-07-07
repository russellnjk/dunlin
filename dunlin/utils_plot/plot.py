import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy             as np 
from matplotlib          import get_backend

import dunlin.utils               as ut
import dunlin.utils_plot.keywords as upk

###############################################################################
#Utility Functions for Plotting 
###############################################################################
def check_skip(skip, c):
    '''Check if scenario c is in skip or if `skip(c)` evaluates to True. 
    '''
    
    if skip is None:
        return False
    elif callable(skip):
        return skip(c)
    else:
        return c in skip

def set_title(ax, title, ref, variable, scenario):
    '''Utility function for setting Axes object title.
    '''
    title = upk.recursive_get(title, variable, scenario)
    
    if title is None:
        return 
    elif callable(title):
        title_ = title(ref=ref, variable=variable, scenario=scenario)
    else:
        title_ = str(title).format(ref=ref, variable=variable, scenario=scenario)
    
    ax.set_title(title_)
    
def label_ax(ax, x, xlabel, y, ylabel, z=None, zlabel=None):
    if xlabel is None:
        ax.set_xlabel(x)
    elif ut.isdictlike(xlabel):
        xlabel = {**{'xlabel': x}, **xlabel}
        ax.set_xlabel(**xlabel)
    else:
        ax.set_xlabel(xlabel)
    
    if ylabel is None:
        ax.set_ylabel(y)
    elif ut.isdictlike(ylabel):
        ylabel = {**{'ylabel': y}, **ylabel}
        ax.set_ylabel(**ylabel)
    else:
        ax.set_ylabel(ylabel)

    if z is None:
        return
    
    if zlabel is None:
        ax.set_zlabel(z)
    elif ut.isdictlike(ylabel):
        zlabel = {**{'zlabel': z}, **zlabel}
        ax.set_zlabel(**zlabel)
    else:
        ax.set_zlabel(zlabel)
    
def gridspec(rows, cols, subplot_args, lb=-2, ub=4, nbins=4, title='', fullscreen=True, 
             left=None, bottom=None, right=None, top=None, wspace=None, hspace=None, 
             **fig_args
             ):
    '''
    subplot_args format: y0, y1, x0, x1, kw
    '''
    fig = plt.figure(**fig_args)
    gs  = fig.add_gridspec(rows, cols)
    
    for arg in subplot_args:
        try:
            y0, y1, x0, x1, kw = arg
            
        except:
            y0, y1, x0, x1 = arg
            kw             = {}
        
        s  = (slice(y0, y1), slice(x0, x1) )
        ax = fig.add_subplot(gs[s], **kw)
        scilimit(ax, lb, ub, nbins)
       
    if not fig_args.get('figsize'):
        fs(fig)
        
    fig.suptitle(**title) if hasattr(title, 'items') else fig.suptitle(title)
    fig.subplots_adjust(left, bottom, right, top, wspace, hspace)
    return fig, fig.axes

def figure(rows=1, cols=1, n=None, lb=-2, ub=4, nbins=4, title='', fullscreen=True, 
           left=None, bottom=None, right=None, top=None, wspace=None, hspace=None,
           tight_layout=False,
           **fig_args
           ):
    fig = plt.figure(**fig_args)
    n   = rows*cols if n is None else n
    
    if type(n) == int:
        to_iter = [{}]*n 
    elif hasattr(n, '__iter__'):
        to_iter = n
    else:
        raise ValueError('n must be an integer or an iterable of keyword arguments for each Axes.')
        
    for i, ax_args in enumerate(to_iter):
        ax       = fig.add_subplot(rows, cols, i+1, **ax_args)
        scilimit(ax, lb, ub, nbins)
    
    if not fig_args.get('figsize'):
        fs(fig)
        
    fig.suptitle(**title) if hasattr(title, 'items') else fig.suptitle(title)
    fig.subplots_adjust(left, bottom, right, top, wspace, hspace)
    
    if tight_layout:
        fig.tight_layout()
        
    return fig, fig.axes

def fs(figure):
    '''
    :meta private:
    '''
    try:
        plt.figure(figure.number)
        backend   = get_backend()
        manager   = plt.get_current_fig_manager()
        
        if backend == 'TkAgg':
            manager.resize(*manager.window.maxsize())
        
        elif backend == 'Qt5Agg' or backend == 'Qt4Agg': 
            manager.window.showMaximized()
        
        else:
            manager.frame.Maximize(True)
        plt.pause(0.03)
    except:
        pass
    return figure

###############################################################################
#Axes
###############################################################################
def make_AX(plot_index):
    '''
    :meta private:
    '''
    #plot_index = {model_key: variables}
    figs = []
    AX   = {}
    if len(plot_index) == 1:
        model_key, variables = next(iter(plot_index.items()))
        
        if len(variables) <= 4:
            rows, cols = len(variables), 1
            fig, AX_   = figure(rows, cols, len(variables))
            
            [ax.set_title(variable) for ax, variable in zip(AX_, variables)]
            figs.append(fig)
            AX = {model_key: dict(zip(variables, AX_))}
        
        elif len(variables) <= 10:
            rows, cols = 2, len(variables)//2 + len(variables)%2
            fig, AX_   = figure(rows, cols, len(variables))
            
            [ax.set_title(variable) for ax, variable in zip(AX_, variables)]

            figs.append(fig)
            AX = {model_key: dict(zip(variables, AX_))}
        
        else:
            AX_list = []
            for i in range(0, len(variables), 10):
                variables_ = variables[i:i+10]
                rows, cols = 2, len(variables_)//2
                fig, AX_   = figure(rows, cols, len(variables_))
                
                [ax.set_title(variable) for ax, variable in zip(AX_, variables_)]

                figs.append(fig)
                AX_list += AX_
                
            AX = {model_key: dict(zip(variables, AX_list))}

    else:
        n_ax   = {}
    
        for model_key, variables in plot_index.items():
            for n, variable in enumerate(variables):
                n_ax.setdefault(n, []).append(variable)
            
        for n, variables in n_ax.items():
            if len(variables) < 4:
                rows, cols = 1, len(variables)
            else:
                sqrt       = len(variables)**0.5
                rows, cols = int(sqrt), int(np.ceil(sqrt)) 
            
            fig, AX_ = figure(rows, cols, len(variables))
            figs.append(fig)
            
            for ax, variable, model_key in zip(AX_, variables, plot_index):
                ax.set_title(str(model_key) + ' ' + str(variable))
                AX.setdefault(model_key, {})[variable] = ax
    
    return figs, AX

###############################################################################
#Axes Formatting
###############################################################################
def wrap_axfig(func):
    def helper(AX, *args, **kwargs):
        if issubclass(type(AX), plt.Figure):
            [helper(ax, *args, **kwargs) for ax in AX.axes]
        elif issubclass(type(AX), dict):
            [helper(value, *args, **kwargs) for value in AX.values()]
        elif issubclass(type(AX), (list, tuple, set)):
            [helper(ax, *args, **kwargs) for ax in AX]
        else:
            func(AX, *args, **kwargs)
    return helper
        
@wrap_axfig    
def scilimit(ax, lb=-2, ub=4, nbins=4):
    ax.ticklabel_format(style='sci', scilimits=(lb, ub))
    ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=4))
    
@wrap_axfig   
def truncate_axis(ax, axis='x', start=0, stop=None):
    if axis == 'x':
        ax.set_xlim(start, stop)
    elif axis == 'y':
        ax.set_ylim(start, stop)
    else:
        raise ValueError('Invalid axis argument given for truncate_axis. Argument "axis" must be "x" or "y"')
        
###############################################################################
#Figure Export
###############################################################################
def save_figs(figs, template, *args):
    for i, fig in enumerate(figs):
        if args:
            filename = template.format(*args, i)
        else:
            filename = template.format(i)
        fig.savefig(filename)