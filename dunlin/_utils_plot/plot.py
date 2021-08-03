import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy             as np 
import seaborn           as sns
from matplotlib          import get_backend


###############################################################################
#Globals
###############################################################################


#Refer for details: https://xkcd.com/color/rgb/
colors = sns.colors.xkcd_rgb

def light_palette(color, n_colors, *args, **kwargs):
    global colors
    if type(color) == str:
        color = colors[color]
        
    palette = sns.light_palette(color, n_colors+2, *args, **kwargs)
    palette = palette[1:len(palette)-1]
    return palette

def dark_palette(color, n_colors, *args, **kwargs):
    global colors
    if type(color) == str:
        color = colors[color]
        
    palette = sns.dark_palette(color, n_colors+2, *args, **kwargs)
    palette = palette[1:len(palette)-1]
    return palette

#Refer for details: https://seaborn.pydata.org/tutorial/color_palettes.html
palette_types = {'color'    : sns.color_palette,
                 'light'    : light_palette,
                 'dark'     : dark_palette,
                 'diverging': sns.diverging_palette,
                 'cubehelix': sns.cubehelix_palette
                 }    

###############################################################################
#Utility Functions for Plotting 
###############################################################################
def recursive_get(dct, *keys):
    if type(dct) != dict:
        return dct
    
    result = dct.get(keys[0], None)
    
    if type(result) == dict:
        if len(keys) == 1:
            raise ValueError(f'The dictionary is too deeply nested. Check the number of levels: {dct}')
        return recursive_get(result, *keys[1:])
    else:
        return result

def gridspec(rows, cols, *subplot_args):
    fig = plt.figure()
    gs  = fig.add_gridspec(rows, cols)
    for arg in subplot_args:
        
        s = (slice(*arg[:2]), slice(*arg[2:]) )
        fig.add_subplot(gs[s])
    fs(fig)
    return fig, fig.axes

def figure(rows, cols, n=None):
    fig = plt.figure()
    n   = rows*cols if n is None else n
    AX  = [fig.add_subplot(rows, cols, i+1) for i in range(n)]
    fs(fig)
    return fig, AX

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
def scilimit(ax, lb=-2, ub=3, nbins=4):
    ax.ticklabel_format(style='sci', scilimits=(-2, 3))
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
            filename = template
        fig.savefig(filename)