import matplotlib.axes as axes
import numpy           as np
import pandas          as pd
import seaborn         as sns
from collections            import namedtuple
from matplotlib.collections import LineCollection
from numbers                import Number 
from typing                 import Any, Callable
  
import dunlin.utils_plot        as upp

#Future work: Statistical analysis and MCMC diagnostics

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
    def merge(cls, traces) -> 'Trace':
        samples         = []
        objective       = []
        context         = []
        raw             = []
        expected        = None
        free_parameters = None
        ref             = None
        trace_args      = None
        
        for i, trace in enumerate(traces):
            if type(trace) != cls:
                msg = 'Expected a {cls.__name__}. Received {type(trace)}.'
                raise TypeError(msg)
            
            samples.append(trace.samples)
            objective.append(trace.objective)
            context.append(trace.context)
            raw.append(trace.raw)
            
            if i == 0:
                expected = set(trace.free_parameters)
            else:
                received   = set(trace.free_parameters)
                missing    = expected - received
                unexpected = received - expected
                if missing:
                    msg = f'Trace {i} is missing {missing}.'
                    raise ValueError(msg)
                if unexpected:
                    msg = f'Trace {i} contains unpexpected parameters {unexpected}.'
                    raise ValueError(msg)
                
            if i == len(traces)-1:
                free_parameters = trace.free_parameters
                ref             = trace.ref
                trace_args      = trace.trace_args 
                
        samples   = pd.concat(samples, ignore_index=True)
        objective = pd.concat(objective, ignore_index=True)
        context   = pd.concat(context, ignore_index=True)
        
        new_obj = cls(free_parameters, 
                      samples, 
                      objective, 
                      context, 
                      raw, 
                      ref, 
                      **trace_args
                      )
        
        return new_obj
    
    def __init__(self, 
                 free_parameters : dict[str, dict],
                 samples         : list[np.ndarray],
                 objective       : list[Number],
                 context         : list[Any],
                 raw             : Any, 
                 ref             : str      = None, 
                 **trace_args   
                 ):
        self.ref             = ref
        self.free_parameters = free_parameters
        self.samples         = pd.DataFrame(samples, columns=list(free_parameters))
        self.objective       = pd.Series(objective, name='objective')
        self.context         = pd.Series(context, name='context')
        self.raw             = raw
        self.trace_args      = {} if trace_args is None else trace_args
        
        
        #For caching
        self._sorted_df: pd.DataFrame = None
        self._posterior: pd.Series    = None
        self._skew     : pd.Series    = None
            
    ###########################################################################
    #Combining Traces
    ###########################################################################     
    def __add__(self, other):
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
    def sorted_df(self) -> pd.DataFrame:
        if self._sorted_df is None:
            sorted_objective = self.objective.sort_values(ascending = True, 
                                                          inplace   = False
                                                          )
            index            = sorted_objective.index
            sorted_samples   = self.samples.loc[index]
            self._sorted_df  = sorted_samples
            
        return self._sorted_df
        
    @property
    def posterior(self) -> pd.Series:
        if self._posterior is None:
            self._posterior = -self.objective
        return self._posterior
    
    @property
    def p(self) -> pd.Series:
        return self.posterior
    
    @property
    def o(self) -> pd.Series:
        return self.other
    
    @property
    def loc(self):
        return self.samples.loc
    
    @property
    def iloc(self):
        return self.samples.iloc
    
    def __getitem__(self, key):
        if key in {'objective', 'posterior', 'context'}:
            return getattr(self, key)
        else:    
            return self.samples[key]
    
    def __len__(self) -> int:
        return len(self.data)
    
    ###########################################################################
    #Analysis
    ###########################################################################     
    def get(self, 
            n           : int|list[int] = 0, 
            sort        : bool = True, 
            ) -> dict|dict[int, dict]:
        if type(n) == list:
            return {i: self.get(i, sort) for i in n}
        elif type(n) != int:
            msg = f'Argument n must be an integer. Received {type(n)}.'
            raise TypeError(msg)
            
        if sort:
            df = self.sorted_df
        else:
            df = self.samples
        
        row       = df.iloc[n]
        idx       = row.name
        objective = self.objective.loc[idx]
        posterior = -objective
        context   = self.context.loc[idx]
        
        result = {'sample'    : row,
                  'objective' : objective,
                  'posterior' : posterior,
                  'context'   : context
                  }
        
        return result
    
    @property
    def best(self):
        return self.get(0, sort=True, reconstruct=True)
    
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
    
    @property
    def skew(self) -> pd.Series:
        if self._skew is None:
            self._skew = self.samples.skew(axis=0)
        return self._skew
    
    ###########################################################################
    #Trace Plotting
    ###########################################################################     
    def _plot_helper(self, 
                     ax      : axes.Axes, 
                     x       : str, 
                     y       : str|None, 
                     default : dict, 
                     kwargs  : dict
                     ) -> tuple:
        if y is None:
            var = x
        else:
            var = x, y
        
        label      = lambda ref, var: '{} {}'.format(ref, var)
        default    = {**default, 'label': label}
        sub_args   = {'ref': self.ref, 'var': var}
        converters = {'color': upp.get_color}
        kwargs     = upp.process_kwargs(kwargs, 
                                        [var], 
                                        default    = {**default, 'label': label},
                                        sub_args   = sub_args, 
                                        converters = converters
                                        )
        
        
        def get_scale(x) -> str:
            scale = self.free_parameters.get(x, {}).get('scale', 'lin')
            scale = self.scale_types[scale]
            return scale
        
        if y is None:
            scale = get_scale(x)
            ax.set_yscale(scale)
            ax.set_ylabel(x)
            y_vals = self[x].values
            
            return y_vals, kwargs
        else:
            scale = get_scale(y)
            ax.set_yscale(scale)
            ax.set_ylabel(y)
            y_vals = self[y].values

            scale = get_scale(x)
            ax.set_yscale(scale)
            x_vals = self[x].values
            ax.set_xlabel(x)
        
        return x_vals, y_vals, kwargs
        
    def plot_kde(self, 
                 ax        : axes.Axes, 
                 parameter : str|tuple[str, str],
                 **kwargs
                 ) -> axes.Axes:
        
        default = {**self.trace_args.get('kde', {}), 
                   'gridsize' : 100
                   }
        
        match parameter:
            case str(x):
                y = None
                
                y_vals, kwargs = self._plot_helper(ax, x, y, default, kwargs)
                
                return sns.kdeplot(x=y_vals, ax=ax, **kwargs)
            
            case [str(x), str(y)]:
                x_vals, y_vals, kwargs = self._plot_helper(ax, x, y, default, kwargs)
                
                return sns.kdeplot(x=x_vals, y=y_vals, ax=ax, **kwargs)
            
            case _:
                msg = f'Could not parse the parameter argument {parameter}.'
                raise ValueError(msg)
    
    def plot_histogram(self, 
                       ax        : axes.Axes, 
                       parameter : str|tuple[str, str],
                       **kwargs
                       ) -> axes.Axes:
        
        default = {'kde_kws': self.trace_args.get('kde', {}),
                   **self.trace_args.get('hist', {}), 
                   }
        
        match parameter:
            case str(x):
                y = None
                
                y_vals, kwargs = self._plot_helper(ax, x, y, default, kwargs)
                
                return sns.histplot(x=y_vals, ax=ax, **kwargs)
            
            case [str(x), str(y)]:
                x_vals, y_vals, kwargs = self._plot_helper(ax, x, y, default, kwargs)
                
                return sns.histplot(x=x_vals, y=y_vals, ax=ax, **kwargs)
            
            case _:
                msg = f'Could not parse the parameter argument {parameter}.'
                raise ValueError(msg)
        
    def plot_steps(self, 
                   ax        : axes.Axes, 
                   parameter : str|tuple[str, str],
                   **kwargs
                   ) -> axes.Axes:
        default = {'marker'          : '+', 
                   'markersize'      : 10,
                   'markeredgewidth' : 3,
                   'linestyle'       : 'None'
                   }
        
        match parameter:
            case str(x):
                y = None
                
                y_vals, kwargs = self._plot_helper(ax, x, y, default, kwargs)
                x_vals         = np.arange(len(y_vals))
                
                if 'colors' in kwargs:
                    kwargs.pop('color', None)
                    stacked  = np.stack([x_vals, y_vals], axis=1)
                    n        = len(kwargs['colors'])
                    d        = int(len(stacked) / n + 1)
                    segments = [stacked[i*d:(i+1)*d+1] for i in range(n)]
                    lines    = LineCollection(segments, **kwargs)
                    result   = ax.add_collection(collection=lines)
                    
                    ax.autoscale()
                    return result
                else:
                    return ax.plot(y_vals, **kwargs)
            
            case [str(x), str(y)]:
                x_vals, y_vals, kwargs = self._plot_helper(ax, x, y, default, kwargs)
                
                if 'colors' in kwargs:
                    kwargs.pop('color', None)
                    stacked  = np.stack([x_vals, y_vals], axis=1)
                    n        = len(kwargs['colors'])
                    d        = int(len(stacked) / n + 1)
                    segments = [stacked[i*d:(i+1)*d+1] for i in range(n)]
                    lines    = LineCollection(segments, **kwargs)
                    result   = ax.add_collection(collection=lines)
                    
                    ax.autoscale()
                    return result
                else:
                    return ax.plot(x_vals, y_vals, **kwargs)
            
            case _:
                msg = f'Could not parse the parameter argument {parameter}.'
                raise ValueError(msg)
    
    ###########################################################################
    #Export
    ###########################################################################
    def to_excel(self, 
                 filename: str|pd.ExcelWriter, 
                 **kwargs
                 ) -> None:
        df = pd.concat([self.samples, self.objective, self.context], axis=1)
        
        return df.to_excel(filename, **kwargs)

###############################################################################
#Trace Analysis
############################################################################### 
def calculate_r_hat(traces):
    if len(traces) < 2:
        raise ValueError('Expected at least 2 traces.')
    
    raise NotImplementedError()


