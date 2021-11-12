###############################################################################
#Non-Standard Imports
###############################################################################
from . import _utils_plot  as upp

###############################################################################
#Globals
###############################################################################
colors        = upp.colors
# palette_types = upp.palette_types

###############################################################################
#Main Algorithms
###############################################################################
def simulate_and_plot_model(model, AX, **sim_args):
    srs = simulate_model(model)
    return plot_simresults(srs, AX, **sim_args)
    
def simulate_model(model, *args, **kwargs):
    simresults = {}
    for intresult in model(*args, **kwargs):
        key             = intresult.scenario
        sim_args        = getattr(model, 'sim_args', {})
        simresults[key] = SimResult(intresult, sim_args)
    
    return simresults

def plot_simresults(simresults, AX, repeat_labels=False, **sim_args):
    plots       = {}
    seen_labels = set()
    for scenario, sr in simresults.items():
        for var, ax_ in AX.items():
            
            #Get axes and plot
            ax   = upp.recursive_get(ax_, scenario) 
            plot = sr.plot(ax, var, **sim_args)
            
            plots.setdefault(var, {})[scenario] = plot
            
            #Reassign label to prevent repeats
            temp = [plot] if hasattr(plot, 'get_label') else plot
            for x in temp:
                label = x.get_label()
                l_key = (ax, label)
                
                if l_key in seen_labels and not repeat_labels:
                    x.set_label('_nolabel')
                else:
                    seen_labels.add(l_key)
    return plots

###############################################################################
#SimResult Class
###############################################################################
class SimResult:
    '''
    Hierarchy: Direct -> exv -> sim_args
    '''
    ###########################################################################
    #Instantiators
    ###########################################################################
    def __init__(self, intresult, sim_args):
        self.intresult = intresult
        self.line_args = sim_args.get('line_args', {'label': 'scenario'})
        self.bar_args  = sim_args.get('bar_args',  {'label': 'scenario'})
        self.hist_args = sim_args.get('hist_args', {'label': 'scenario'})
        self.colors    = sim_args.get('colors', {})
        
    ###########################################################################
    #Accessors and Representation
    ###########################################################################
    def __getitem__(self, key):
        return self.intresult[key]
    
    def __str__(self):
        return f'{type(self).__name__}({self.intresult.model_key}|{self.intresult.scenario})'
    
    def __repr__(self):
        return self.__str__()
    
    ###########################################################################
    #Plotters
    ###########################################################################
    def plot(self, ax, var, **sim_args):
        int_data = self[var]
        
        if type(int_data) == dict:
            plot_type   = int_data.get('plot_type', 'line')
            int_data    = {k: v for k, v in int_data.items() if k != 'plot_type'}
            key         = plot_type + '_args'
            merged_args = {**int_data, **sim_args.get(key, {})}
            
            if plot_type == 'bar':
                ax.set_title(var)
                return self.plot_bar(var, ax, **merged_args)
            elif plot_type == 'line':
                ax.set_title(var)
                return self.plot_line(var, ax, **merged_args)
            if plot_type == 'hist':
                ax.set_title(var)
                return self.plot_hist(var, ax, **merged_args)
            else:
                raise NotImplementedError(f'No implementation for plot_type {plot_type}.')
        
        elif type(var) in [tuple, list]:
            
            if len(var) == 1:
                ax.set_title(var[0])
                return self.plot_line(var, ax, self['t'], int_data, **sim_args.get('line_args', {}))
            elif len(var) == 2:
                ax.set_title(f'{var[0]} vs {var[1]}')
                return self.plot_line(var, ax, int_data[0], int_data[1], **sim_args.get('line_args', {}))
            else:
                raise NotImplementedError(f'Cannot plot {var}. Length must be one or two.')
            
        elif type(var) == str:
            ax.set_title(var)
            return self.plot_line(var, ax, self['t'], int_data, **sim_args.get('line_args', {}))

        else:
            raise NotImplementedError(f'No implementation for {var}')
            
    def plot_line(self, var, ax, x, y, **kwargs):
        kwargs = self._process_plot_args(var, self.line_args, kwargs)
        result = ax.plot(x, y, **kwargs)
        
        return result
    
    def plot_bar(self, var, ax, x, height, **kwargs):
        kwargs = self._process_plot_args(var, self.bar_args, kwargs)
        result = ax.bar(x, height, **kwargs)
        
        ax.set_xticks(range(len(x)))
        
        return result
    
    def plot_hist(self, var, ax, x, **kwargs):
        kwargs = self._process_plot_args(var, self.hist_args, kwargs)
        result = ax.hist(x, **kwargs)
        
        return result
    
    def _process_plot_args(self, var, self_args, ext_args):
        scenario  = self.intresult.scenario
        model_key = self.intresult.model_key
        
        def recurse(dct):
            if dct is None:
                return {}
            return {k: upp.recursive_get(v, scenario, var) for k, v in dct.items()}
            
        self_args = recurse(self_args)
        ext_args  = recurse(ext_args) 
        
        kwargs = {**self_args, **ext_args}
        
        #Process label
        label_scheme   = kwargs.get('label', 'scenario')
        if label_scheme == 'scenario':
            label = f'{scenario}'
        elif label_scheme == 'model_key':
            label = f'{model_key}'
        elif label_scheme == 'state':
            label = f'{var}'
        elif label_scheme == 'model_key, scenario':
            label = f'{model_key}, {scenario}'
        elif label_scheme == 'state, scenario':    
            label = f'{var}, {scenario}'
        elif label_scheme == 'model_key, state':    
            label = f'{var}, {var}'
        elif label_scheme == 'model_key, state, scenario':
            label = f'{model_key}, {var}, {scenario}'
        else:
            label = label_scheme
        
        kwargs['label'] = label
        
        #Process color
        color = kwargs.get('color')
        if type(color) == str:
            kwargs['color'] = colors[color]

        return kwargs
