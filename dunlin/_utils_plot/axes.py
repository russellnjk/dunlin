import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy             as np 
import seaborn           as sns
from matplotlib          import get_backend


###############################################################################
#Globals
###############################################################################
#Refer for details: https://seaborn.pydata.org/tutorial/color_palettes.html
palette_types = {'color':     lambda n_colors, **kwargs : sns.color_palette(n_colors=n_colors,     **{**{'palette': 'muted'}, **kwargs}),
                 'light':     lambda n_colors, **kwargs : sns.light_palette(n_colors=n_colors+2,   **{**{'color':'steel'}, **kwargs})[2:],
                 'dark' :     lambda n_colors, **kwargs : sns.dark_palette( n_colors=n_colors+2,   **{**{'color':'steel'}, **kwargs})[2:],
                 'diverging': lambda n_colors, **kwargs : sns.diverging_palette(n=n_colors,        **{**{'h_pos': 250, 'h_neg':15}, **kwargs}),
                 'cubehelix': lambda n_colors, **kwargs : sns.cubehelix_palette(n_colors=n_colors, **kwargs),
                 }    

#Refer for details: https://xkcd.com/color/rgb/
colors = sns.colors.xkcd_rgb
    
###############################################################################
#Nested Args
###############################################################################
def parse_recursive(nested_args, *keys, apply=True, name='data', error_if_missing=False):
    if apply:
        # result = {key: recurse(value, *keys, name=name, error_if_missing=error_if_missing) for key, value in nested_args.items()}
        result = {}
        for key, value in nested_args.items():
            temp = recurse(value, *keys, name=name, error_if_missing=error_if_missing)
            if temp is None:
                continue
            else:
                result[key] = temp
        return result
    else:
        return recurse(nested_args, *keys, name=name, error_if_missing=error_if_missing)
    
def recurse(data, *keys, name='data', error_if_missing=True):
    if type(data) == dict and len(keys):
        if keys[0] in data:
            return recurse(data[keys[0]], *keys[1:], name=name, error_if_missing=error_if_missing)
        else:
            if error_if_missing:
                raise Exception('Missing {} for {}'.format(name, keys[0]))
            else:
                return None
    else:
        return data
    
###############################################################################
#Legend
###############################################################################
def apply_legend(AX, legend_args=None):
    if type(AX) == dict:
        [apply_legend(value, parse_recursive(legend_args, key, apply=False) ) for key, value in AX.items()]
    elif legend_args:
        AX.legend(**legend_args)
    else:
        AX.legend()
    
###############################################################################
#Figure
###############################################################################
def figure(rows, cols, n=None):
    n1  = n if n is not None else rows*cols 
    fig = plt.figure()
    if n1 > 1:
        AX  = [fig.add_subplot(rows, cols, i+1) for i in range(n1)]
    else:
        AX = [fig.add_subplot(1, 1, 1)]
    
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
def truncate_time(ax, start=0, stop=None):
    ax.set_xlim(start, stop)

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
        
if __name__ == '__main__':
    # #Test Axes generation
    # plot_index = {1: [1, 2, 3],
    #               2: [1, 2],
    #               }
    
    # figs, AX = make_AX(plot_index)
    # assert len(figs) == 3
    
    # plot_index = {1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    #               }
    
    # figs, AX = make_AX(plot_index)
    # assert len(figs) == 2
    pass