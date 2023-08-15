import matplotlib.axes as axes
import numpy           as np
import pandas          as pd
from matplotlib.collections import LineCollection
from numbers                import Number
from typing                 import Any

###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin.utils          as ut
import dunlin.comp           as cmp
import dunlin.datastructures as dst
import dunlin.ode.ode_coder  as odc
import dunlin.utils_plot     as upp
from .basemodel import BaseModel

#Type hints
Scenario = str|Number

def is_scenario(c: Any) -> bool:
    try:
        if isinstance(c, Scenario):
            return True
        elif all(isinstance(i, Scenario) for i in c):
            return True
        else:
            return False
    except:
        return False
    
class ODEModel(BaseModel):
    '''
    After instantiation, the model's `external` attribute can be modified 
    '''
    
    #Class attributes not defined in the parent class
    #Refer to the parent class for other class attributes as well as required 
    #attributes
    _dtype        = 'ode'
    default_tspan = np.linspace(0, 1000, 21)
    
    @classmethod
    def from_data(cls, all_data: dict, ref: str) -> 'ODEModel':
        flattened = cmp.flatten_model(all_data, ref, dst.ODEModelData.required_fields)
        return cls(**flattened)
    
    def __init__(self, 
                 ref          : str, 
                 states       : pd.DataFrame, 
                 parameters   : pd.DataFrame, 
                 functions    : dict = None, 
                 variables    : dict = None, 
                 reactions    : dict = None, 
                 rates        : dict = None, 
                 events       : dict = None, 
                 tspans       : dict = None,
                 compartments : dict = None, 
                 int_args     : dict = None, 
                 sim_args     : dict = None, 
                 optim_args   : dict = None, 
                 trace_args   : dict = None,
                 data_args    : dict = None, 
                 meta         : dict = None,
                 dtype        : str  = 'ode',
                 ):
        
        model_data = dst.ODEModelData(ref        = ref, 
                                      tspans     = tspans, 
                                      states     = states,
                                      parameters = parameters,
                                      functions  = functions,
                                      variables  = variables,
                                      reactions  = reactions,
                                      rates      = rates,
                                      events     = events,
                                      meta       = meta
                                      )
        
        (rhs0, rhs1), (rhsdct0, rhsdct1), event_objects = odc.make_ode_callables(model_data)
        
        super().__init__(model_data = model_data, 
                         ref        = ref, 
                         tspans     = tspans, 
                         int_args   = int_args, 
                         dtype      = dtype, 
                         events     = event_objects
                         )
        
        self._rhs_functions    = rhs0, rhs1
        self._rhsdct_functions = rhsdct0, rhsdct1
        
        #Viewable but not editable by front-end users
        #To be set once during instantiation and subsequently locked
        self.states     = tuple(model_data.states)
        self.parameters = tuple(model_data.parameters)
        self.functions  = tuple(model_data.functions)
        self.variables  = tuple(model_data.variables)
        self.reactions  = tuple(model_data.reactions)
        self.events     = tuple(model_data.events)
        self.meta       = model_data.meta
        
        #For back-end only
        #For storing state and parameters
        self.state_df     = states
        self.parameter_df = parameters
        
        #Specific to this class
        self.trace_args = {} if trace_args is None else trace_args 
        self.optim_args = {} if optim_args is None else optim_args
        self.sim_args   = {} if sim_args   is None else sim_args
        self.data_args  = {} if data_args  is None else data_args
        
    def _convert_call(self, 
                      t : np.ndarray, 
                      y : np.ndarray, 
                      p : np.ndarray,
                      c : Scenario
                      ) -> 'ODEResult':
        return ODEResult(t, y, p, c, self)
    
    def _convert_integrate(self, 
                           scenario2intresult: dict[Scenario, 'ODEResult']
                           ) -> 'ODEResultDict':
        return ODEResultDict(scenario2intresult, self)
        
###############################################################################
#Integration Results
###############################################################################
class ODEResult:
    ###########################################################################
    #Instantiators
    ###########################################################################
    def __init__(self, 
                 t     : np.ndarray, 
                 y     : np.ndarray, 
                 p     : np.ndarray, 
                 c     : Scenario,
                 model : ODEModel
                 ):
        
        namespace = (model.states 
                     + model.parameters 
                     + model.variables 
                     + model.reactions
                     + tuple([ut.diff(x) for x in model.states])
                     + tuple(model.externals)
                     )
        namespace = frozenset(namespace)
        
        self.ref         = model.ref
        self.t           = t
        self.y           = dict(zip(model.states,     y))
        self.p           = dict(zip(model.parameters, p)) 
        self.scenario    = c
        self.internals   = {}
        self.externals   = {}
        self.args        = t, y, p
        self.rhsdct      = model.rhsdct
        self.rhsexternal = model.externals
        self.line_args   = model.sim_args.get('line_args', {})
        
        #Keep track of names for lazy evaluation
        self.internal_names = frozenset(model.variables  
                                        + model.reactions 
                                        + tuple([ut.diff(x) for x in model.states])
                                        )
        self.external_names = frozenset(model.externals.keys())
        
    ###########################################################################
    #Accessors/Lazy Evaluators
    ###########################################################################
    def __getitem__(self, name: str):
        if ut.islistlike(name):
            return [self[i] for i in name]
        elif name == 'time':
            return self.t
        elif name in self.y:
            return self.y[name]
        elif name in self.p:
            return self.p[name]
        elif name in self.internal_names:
            if not self.internals:
                self.internals.update( self.rhsdct(*self.args) )
            return self.internals[name]
        elif name in self.external_names:
            if name not in self.externals:
                self.externals[name] = self.rhsexternal[name](self)
            return self.externals[name]
        else:
            raise KeyError(name)
        
    def get(self, name: str, default: Any=None) -> Any:
        if ut.islistlike(name):
            return [self.get(i) for i in name]
        
        try:
            return self[name]
        except KeyError:
            return default
    
    def __contains__(self, var):
        return var in self.namespace
            
    ###########################################################################
    #Representation
    ###########################################################################
    def __str__(self):
        s = tuple(self.ref)
        c = self.scenario
        return f'{type(self).__name__}{s} {c}'
    
    def __repr__(self):
        return str(self)
    
    ###########################################################################
    #Plotting
    ###########################################################################
    def plot_line(self,
                  ax     : axes.Axes,
                  var    : str|tuple[str, str],
                  **kwargs
                  ) -> axes.Axes:
        
        match var:
            case str(y):
                x_vals = self['time']
                y_vals = self[y]
                
            case [str(x), str(y)]:
                x_vals = self[x]
                y_vals = self[y]
            
            case _:
                msg = f'Could not parse the var argument {var}.'
                raise ValueError(msg)
        
        label      = lambda ref, scenario, var: '{} {} {}'.format(ref, scenario, var)
        default    = {'label': label, **self.line_args}
        sub_args   = {'ref': self.ref, 'scenario': self.scenario, 'var': var}
        converters = {'color'  : upp.get_color,
                      'colors' : upp.get_colors
                      }
        kwargs     = upp.process_kwargs(kwargs, 
                                        [self.scenario, var], 
                                        default    = default,
                                        sub_args   = sub_args, 
                                        converters = converters
                                        )
        
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
        
###############################################################################
#SimResult Class
###############################################################################
class ODEResultDict:
    def __init__(self, 
                 scenario2intresult : dict[Scenario, ODEResult],
                 model              : ODEModel
                 ):
        self.scenario2intresult = scenario2intresult
    
    def __getitem__(self, key: Scenario) -> ODEResult:
        return self.scenario2intresult[key]
    
    def plot_line(self,
                  ax        : axes.Axes,
                  var       : str|tuple[str, str],
                  scenarios : Scenario|list[Scenario] = None,
                  **kwargs
                  ) -> axes.Axes:
        
        match scenarios:
            case None:
                scenarios = set(self.scenario2intresult)
            case [*scenarios]:
                scenarios = set(scenarios)
            case c if isinstance(c, Scenario):
                scenarios = {c}
            case _:
                msg = ''
                raise ValueError(msg)
        
        result = {}
        for scenario, oderesult in self.scenario2intresult.items():
            if scenario not in scenarios:
                continue
            
            result[scenario] = oderesult.plot_line(ax, var, **kwargs)
        
        return result
     