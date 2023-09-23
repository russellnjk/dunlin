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
Scenario = str|Number|tuple[str|Number]

def is_scenario(c: Any) -> bool:
    try:
        if isinstance(c, (str, Number)):
            return True
        elif all(isinstance(i, (str, Number)) for i in c):
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
                 states       : dict|pd.DataFrame, 
                 parameters   : dict|pd.DataFrame, 
                 functions    : dict = None, 
                 variables    : dict = None, 
                 reactions    : dict = None, 
                 rates        : dict = None, 
                 events       : dict = None, 
                 tspans       : dict = None,
                 domain_types : dict = None, 
                 int_args     : dict = None, 
                 sim_args     : dict = None, 
                 opt_args     : dict = None, 
                 trace_args   : dict = None,
                 data_args    : dict = None,
                 meta         : dict = None,
                 dtype        : str  = 'ode',
                 ):
        
        #Parse the data using the datastructures submodule
        model_data = dst.ODEModelData(ref          = ref, 
                                      tspans       = tspans, 
                                      states       = states,
                                      parameters   = parameters,
                                      functions    = functions,
                                      variables    = variables,
                                      reactions    = reactions,
                                      rates        = rates,
                                      events       = events,
                                      meta         = meta,
                                      domain_types = domain_types, 
                                      int_args     = int_args, 
                                      sim_args     = sim_args, 
                                      opt_args     = opt_args, 
                                      trace_args   = trace_args,
                                      data_args    = data_args
                                      )
        
        #Generate code, functions and events
        (rhs0, rhs1), (rhsdct0, rhsdct1), event_objects = odc.make_ode_callables(model_data)
        
        #Call the parent constructor to save key information
        super().__init__(model_data = model_data, 
                         ref        = ref, 
                         tspans     = tspans, 
                         int_args   = int_args, 
                         dtype      = dtype, 
                         events     = event_objects
                         )
        #Assign rhs 
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
        self.int_args   = {} if model_data.int_args   is None else model_data.int_args
        self.sim_args   = {} if model_data.sim_args   is None else model_data.sim_args
        self.opt_args   = {} if model_data.opt_args   is None else model_data.opt_args
        self.trace_args = {} if model_data.trace_args is None else model_data.trace_args 
        self.data_args  = {} if model_data.data_args is None else model_data.data_args
        
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
        
        #Keep track of names for lazy evaluation
        self.internal_names = frozenset(model.variables  
                                        + model.reactions 
                                        + tuple([ut.diff(x) for x in model.states])
                                        )
        self.external_names = frozenset(model.externals.keys())
        
        
        #For plotting
        label          = lambda ref, scenario, var: '{} {} {}'.format(ref, scenario, var)
        self.line_args = {'label': label, **model.sim_args.get('line_args', {})}
        
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
                  ax             : axes.Axes,
                  var            : str|tuple[str, str],
                  ignore_default : bool = False,
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
        
        default    = {} if ignore_default else self.line_args
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
                scenarios = None
            case scenario if is_scenario(scenario):
                scenarios = {scenario}
            case scenarios if all([is_scenario(c) for c in scenarios]):
                scenarios = set(scenarios)
            case _:
                msg  = 'Unexpected format. The scenarios argument must be '
                msg += 'a scenario or list of scenarios. '
                msg += 'A single scenario is a string, number or tuple of strings/numbers.'
                raise ValueError(msg)
        
        result = {}
        for scenario, intresult in self.scenario2intresult.items():
            if scenarios is not None:
                if scenario not in scenarios:
                    continue
            
            result[scenario] = intresult.plot_line(ax, var, **kwargs)
        
        return result
     