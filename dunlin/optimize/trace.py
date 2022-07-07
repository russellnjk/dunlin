import numpy             as     np
import pandas            as     pd
from collections         import namedtuple
from scipy.stats         import skewtest, kurtosistest

import dunlin.utils             as ut
import dunlin.utils_plot        as upp

#Future work: Statistical analysis and MCMC diagnostics

###############################################################################
#Trace Analysis
############################################################################### 
def calculate_r_hat(traces):
    if len(traces) < 2:
        raise ValueError('Expected at least 2 traces.')
    
    raise NotImplementedError()

def merge_traces(traces):
    return Trace.merge(traces)

###############################################################################
#Trace Class
############################################################################### 
best = namedtuple('BestParameters', 'parameters objective posterior')

class Trace:
    scale_types = {'lin' : 'linear', 'log10': 'log10', 'log': 'log'}
    
    ###########################################################################
    #Constructors
    ###########################################################################  
    @classmethod
    def merge(cls, traces):
        trace_dfs = []
        other     = None
        first     = True
        
        for trace in traces:
            if type(trace) != cls:
                trace = trace.trace
            
            trace_dfs.append(trace.data)
            
            if first:
                free_parameters = trace.free_parameters
                ref             = trace.ref
                trace_args      = trace.trace_args 
                first           = False
            else:
                if list(trace.data.columns) != list(trace_dfs[0].columns):
                    raise ValueError('Could not merge traces as columns were mismatched.')
                    
        trace_df = pd.concat(trace_dfs, ignore_index=True)
        new_obj  = cls(trace_df, other, free_parameters, ref, trace_args)
        
        return new_obj
    
    def __init__(self, trace_df, other, free_parameters, ref=None, trace_args=None):
        self.other           = other
        self._data           = trace_df.astype(np.float64)
        self.trace_args      = {} if trace_args is None else trace_args
        self.ref             = ref
        self.free_parameters = free_parameters
        
        #For caching
        self._sorted_df = None
        
        if set(free_parameters).difference( set(trace_df.columns) ):
            raise ValueError('Free parameters and trace DataFrames do not match.')
            
    ###########################################################################
    #Combining Traces
    ###########################################################################     
    def __add__(self, other):
        if ut.islistlike(other):
            return self.merge([self, *other])
        else:
            return self.merge([self, other])
    
    ###########################################################################
    #Representation
    ###########################################################################     
    def __repr__(self):
        return str(self)
    
    def __str__(self):
        keys = tuple(self.free_parameters.keys())
        name = type(self).__name__
        
        return f'{name}{keys}'
    
    ###########################################################################
    #Access
    ###########################################################################  
    @property
    def trace(self):
        return self
    
    @property
    def data(self):
        return self._data
    
    @property
    def sorted_df(self):
        if self._sorted_df is None:
            self._sorted_df = self.data.sort_values(by='posterior', 
                                                    ascending=False, 
                                                    inplace=False
                                                    )
        
        return self._sorted_df
        
    @property
    def objective(self):
        return self.data['objective']
    
    @property
    def posterior(self):
        return self.data['posterior']
    
    @property
    def context(self):
        return self.data['context']
    
    @property
    def p(self):
        return self.posterior
    
    @property
    def o(self):
        return self.other
    
    @property
    def loc(self):
        return self.data.loc
    
    @property
    def iloc(self):
        return self.data.iloc
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __len__(self):
        return len(self.data)
    
    ###########################################################################
    #Analysis
    ###########################################################################     
    def get_best(self, n=0):
        df        = self.sorted_df
        df        = df.iloc[n]
        posterior = df['posterior']
        objective = df['objective']
        
        if type(df) == pd.Series:
            parameters = df.iloc[:len(self.free_parameters)]
        else:
            parameters = df.iloc[:, :len(self.free_parameters)]
            
        return best(parameters, objective, posterior)
    
    @property
    def best(self):
        df         = self.sorted_df
        series     = df.iloc[0]
        objective  = series['objective']
        posterior  = series['posterior']
        parameters = series.iloc[:len(self.free_parameters)]
        return best(parameters, objective, posterior)
    
    def var(self, last=0.5):
        df  = self.data.iloc[:, :len(self.free_parameters)]
        
        n  = round(len(df.index)*last)
        df = df.iloc[:,-n:]
        var = df.var(axis=0)
            
        return var
    
    def sd(self, last=0.5):
        df  = self.data.iloc[:, :len(self.free_parameters)]
        
        n  = round(len(df.index)*last)
        df = df.iloc[:,-n:]
        sd = df.std(axis=0)
            
        return sd
    
    ###########################################################################
    #Trace Plotting
    ###########################################################################     
    def plot(self, ax, var, plot_type='line', **trace_args):
        if plot_type == 'line':
            if ut.islistlike(var):
                x, y = var
                return self.plot_steps(ax, *var, **trace_args)
            else:
                return self.plot_steps(ax, None, var, **trace_args)
        elif plot_type == 'hist':
            raise NotImplementedError()
        elif plot_type == 'kde':
            raise NotImplementedError()
        else:
            raise ValueError(f'Unrecognized plot_type {plot_type}')
    
    def get_scale(self, x):
        if ut.is_valid_name(x):
            scale = self.free_parameters[x].get('scale', 'lin')
        else:
            scale = 'lin'
        
        scale = self.scale_types[scale]
        return scale
    
    def plot_steps(self, ax, x, y=None, **kwargs):
        ext_args  = kwargs
        self_args = self.trace_args.get('step', {})
        
        if y is None:
            kwargs  = self._recursive_get(self_args, ext_args, x)
        else:
            kwargs  = self._recursive_get(self_args, ext_args, (x, y)) 
            
        kwargs.setdefault('marker', '+')
        kwargs.setdefault('linestyle','None')
        
        if y is None:
            scale = self.get_scale(x)
            ax.set_yscale(scale)
            ax.set_ylabel(x)
            y_vals = self[x].values
            
            return ax.plot(y_vals, **kwargs)
        else:
            scale = self.get_scale(y)
            ax.set_yscale(scale)
            ax.set_ylabel(y)
            y_vals = self[y].values

            scale = self.get_scale(x)
            ax.set_yscale(scale)
            x_vals = self[x].values
            ax.set_xlabel(x)
            
            return ax.plot(x_vals, y_vals, **kwargs)
        
    def _recursive_get(self, self_args, ext_args, var):
        global colors
        
        keys = self.ref, var
        def recurse(dct):
            if dct is None:
                return {}
            return {k: upp.recursive_get(v, *keys) for k, v in dct.items()}
        
        
        self_args = recurse(self_args)
        ext_args  = recurse(ext_args) 
        kwargs    = {**self_args, **ext_args}
        
        #Process special keywords
        label = kwargs.get('label', '{ref}')
        
        if callable(label):
            label = label(ref=self.ref, var=var)
        else:
            label = label.format(ref=self.ref, var=var)
        
        kwargs['label'] = label
        
        #Process color
        color = kwargs.get('color')
        
        if callable(color):
            color = color(ref=self.ref, var=var)
        else:
            color = upp.get_color(color)
        
        kwargs['color'] = color

        return kwargs
            