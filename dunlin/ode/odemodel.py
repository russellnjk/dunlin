import numpy  as np
import pandas as pd
from typing import Any, Union

###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin.utils          as ut
import dunlin.comp           as cmp
import dunlin.datastructures as dst
import dunlin.ode.ode_coder  as odc
import dunlin.ode.event      as oev
import dunlin.ode.ivp        as ivp
import dunlin.utils_plot     as upp
from .basemodel import BaseModel

#Type hints
Scenario = Union[str, float, int, tuple]

class ODEModel(BaseModel):
    '''
    '''
    default_tspan = np.linspace(0, 1000, 21)
    _locked       = ['state_names', 'parameter_names', 'functions', 'reactions', 
                     'variables']
    
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
        super().__init__(model_data, ref, tspans, int_args, dtype)
        
        model_data = self._model_data
        
        (rhs0, rhs1), (rhsdct0, rhsdct1), events = odc.make_ode_callables(model_data)
        
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
        self.optim_args = optim_args
        self.sim_args   = sim_args
        
    def _convert_raw_output(self, t, y, p) -> type:
        return ODEResult(t, y, p, self)
        
    def simulate(self, **kwargs):
        return ODESimResult(self, **kwargs)
        
class ModelMismatchError(Exception):
    def __init__(self, expected, received):
        super().__init__(f'Required keys: {list(expected)}. Recevied: {list(received)}')
        
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
        
        self.t           = t
        self.y           = dict(zip(model.states,     y))
        self.p           = dict(zip(model.parameters, p)) 
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
        
    ###########################################################################
    #Accessors/Lazy Evaluators
    ###########################################################################
    @property
    def eval_extra(self):
        if self._eval_extra is None:
            self._eval_extra = self.extra(*self._args)
        return self._eval_extra
    
    @property
    def eval_dct(self):
        if self._eval_dct is None:
            self._eval_dct = self.dct(*self._args)
        return self._eval_dct
    
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
        s = tuple(self.namespace)
        return f'{type(self).__name__}{s}'
    
    def __repr__(self):
        return self.__str__()
    
###############################################################################
#SimResult Class
###############################################################################
class ODESimResult:
    _line_args = {'label': '{scenario}'}
    _bar_args  = {'width': 0.4, 'bottom': 0}
    
    ###########################################################################
    #Instantiators
    ###########################################################################
    def __init__(self, model, sim_args=None, **kwargs):
        self.intresults      = dict(model.integrate(**kwargs).items())
        self.levels          = list(model.states.index.names) 
        self.ref             = model.ref
        self.variables       = set()
        self.extra_variables = set()
        
        for ir in self.intresults.values():
            self.variables.update(ir.namespace)
            self.extra_variables.update(ir.extra_variables)
        
        if sim_args is not None:
            self.sim_args = sim_args 
        elif model.sim_args is None: 
            self.sim_args = {}
        else:
            self.sim_args = model.sim_args
            
    ###########################################################################
    #Accessors
    ###########################################################################
    def get(self, variable=None, scenario=None, _extract=True):
        if type(variable) == list:
            dct = {}
            for v in variable:
                temp = self.get(v, scenario, False)
                dct.update(temp)
            return dct
        elif type(scenario) == list:
            dct = {}
            for c in scenario:
                temp = self.get(variable, c, False)
                dct.update(temp)
            return dct
        
        
        if variable is not None and variable not in self.variables:
            raise ValueError(f'Unexpected variable: {repr(variable)}')
        elif scenario is not None and scenario not in self.intresults:
            raise ValueError(f'Unexpected scenario: {repr(scenario)}')
        
        dct  = {}
        
        for c, ir in self.intresults.items():
            if not ut.compare_scenarios(c, scenario):
                continue
            
            time = ir['time']

            for v in ir.namespace:
                
                if not ut.compare_variables(v, variable):
                    continue
                
                values = ir[v]
                name   = v, c
                
                if v in ir.extra_variables:
                    dct.setdefault(v, {})[c] = values
                else:
                    series = pd.Series(values, index=time, name=name)
                    dct.setdefault(v, {})[c] = series
        
        if variable is not None and scenario is not None and _extract:
            return dct[variable][scenario]
        else:
            return dct
        
    def __getitem__(self, key):
        if type(key) == tuple:
            if len(key) != 2:
                raise ValueError('Expected a tuple of length 2.')
            variable, scenario = key  
        else:
            variable = key
            scenario = None
            
        return self.get(variable, scenario)
    
    def has(self, variable=None, scenario=None):
        if type(variable) in [list, tuple]:
            return all([self.has(v, scenario) for v in variable])
        elif type(scenario) == list:
            return all([self.has(variable, c) for c in scenario])
        
        if variable is None and scenario is None:
            raise ValueError('variable and scenario cannot both be None.')
        elif variable is None:
            return scenario in self.intresults
        elif scenario is None:
            return variable in self.variables
        else:
            return variable in self.variables and scenario in self.intresults
    
    ###########################################################################
    #Representation
    ###########################################################################
    def __str__(self):
        
        return f'{type(self).__name__}{tuple(self.intresults.keys())}'
    
    def __repr__(self):
        return self.__str__()
    
    ###########################################################################
    #Master Plotting Method
    ###########################################################################
    def plot(self, AX_lst, *args, **sim_args):
        '''Still in the works. Meant to be used for future work on high-level 
        declarative plotting.
        '''
        raise NotImplementedError('Not implemented yet.')
        if not ut.islistlike(AX_lst):
            AX_lst_ = [AX_lst]
        
        result = []
        for ax in AX_lst_:
            plot_type = getattr(ax, 'plot_type', '2D')
            method    = getattr(self, 'plot_'+plot_type, None)
        
            if method is None:
                raise AttributeError(f'No method found for plot_type "{plot_type}"')
            
            kwargs = getattr(ax, 'args', {})
            kwargs = {**kwargs, **sim_args}
            result.append( method(ax, *args, **kwargs) )
        
        if not ut.islistlike(AX_lst):
            return result[0]
        else:
            return result
        
    ###########################################################################
    #Plotting Functions
    ###########################################################################
    def plot_line(self, ax_dct, variable, xlabel=None, ylabel=None, title=None, 
                  skip=None, **line_args
                  ):
        
        #Determine which variables to plot
        if ut.islistlike(variable):
            x, y = variable
        else:
            x, y = 'time', variable
        
        #Set user line_args
        if line_args:
            line_args = {**self.sim_args.get('line_args', {}), **line_args}
        else: 
            line_args = self.sim_args.get('line_args', {})
            
        result      = {}
        #Iterate and plot
        for c, ir in self.intresults.items():
            if upp.check_skip(skip, c):
                continue
            
            #Process the plotting args
            keys       = [c, variable]
            sub_args   = dict(scenario=c, variable=variable, ref=self.ref)
            converters = {'color': upp.get_color}
            line_args_ = upp.process_kwargs(line_args, 
                                            keys, 
                                            default=self._line_args, 
                                            sub_args=sub_args, 
                                            converters=converters
                                            )
            
            #Determine axs
            ax = upp.recursive_get(ax_dct, c)
            
            if ax is None:
                continue
            
            #Plot
            result[c] = ax.plot(ir[x], ir[y], **line_args_)           
            
            #Label axes
            upp.label_ax(ax, x, xlabel, y, ylabel)
            upp.set_title(ax, title, self.ref, variable, c)
            
        return result
         
    def plot_bar(self, ax, variable, by='scenario', xlabel=None, ylabel=None, 
                 skip=None, horizontal=False, stacked=False, **bar_args
                 ):
        '''For extra only
        '''
        #Determine which variables to plot
        if ut.islistlike(variable):
            variables = list(variable)
        else:
            variables = [variable]
        
        #Create the dataframe
        dct = {}
        for c, ir in self.intresults.items():
            if upp.check_skip(skip, c):
                continue
            
            for v in variables:
                
                if v not in self.extra_variables:
                    msg = f'plot_bar can only be used extra variables. Received {v}'
                    raise ValueError(msg)
                    
                dct.setdefault(v, {})[c] = ir[v]
                
        df = pd.DataFrame(dct)
        
        #Determine how the bars should be grouped
        if by == 'scenario':
            pass
        elif by == 'variable':
            df = df.T
        else:
            df = df.unstack(by).stack(0)
        

        #Prepare the sim args
        def getter(d, f=None):
            if hasattr(d, 'items') and f:
                return {k: f(v) for k, v in d.items()}
            else:
                return d
        
        bar_args   = bar_args if bar_args else self.sim_args.get('bar_args', {})
        keys       = []
        sub_args   = dict(scenario=c, variable=variable, ref=self.ref)
        converters = {'color'     : lambda d: getter(d, upp.get_color),
                      'edgecolor' : lambda d: getter(d, upp.get_color),
                      }
        bar_args_  = upp.process_kwargs(bar_args, 
                                        keys, 
                                        default=self._bar_args, 
                                        sub_args=sub_args, 
                                        converters=converters
                                        )
        
        result = upp.plot_bar(ax, 
                              df, 
                              xlabel, 
                              ylabel, 
                              horizontal=horizontal, 
                              stacked=stacked, 
                              **bar_args_
                              )
        return result
    
