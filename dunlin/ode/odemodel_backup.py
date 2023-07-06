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

#Type hints
Scenario = Union[str, float, int, tuple]

class ODEModel:
    '''
    '''
    default_tspan = np.linspace(0, 1000, 21)
    _locked       = ['state_names', 'parameter_names', 'functions', 'reactions', 
                     'variables']
    
    @classmethod
    def from_data(cls, all_data: dict, ref: str) -> 'ODEModel':
        flattened = cmp.flatten_ode(all_data, ref)
        
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
                 extra        : dict = None, 
                 tspan        : dict =None,
                 compartments : dict = None, 
                 int_args     : dict = None, 
                 sim_args     : dict = None, 
                 optim_args   : dict = None, 
                 data_args    : dict = None, 
                 meta         : dict = None,
                 dtype        : str   = 'ode',
                 **kwargs
                 ):
        
        if dtype != 'ode':
            msg = f'Attempted to instantiate {type(self).__name__} with {dtype} data.'
            raise TypeError(msg)
        
        functions = {} if functions is None else functions
        variables = {} if variables is None else variables
        reactions = {} if reactions is None else reactions
        rates     = {} if rates     is None else rates
        extra     = {} if extra     is None else extra
        
        model_data = dst.ODEModelData(ref, 
                                      states, 
                                      parameters, 
                                      functions, 
                                      variables, 
                                      reactions, 
                                      rates, 
                                      events, 
                                      extra, 
                                      )
        
        ode = odc.make_ode(model_data)
        
        #Assign ode attributes
        self._rhs       = ode.rhs
        self._events    = oev.make_events(ode.rhsevents)
        self._extra     = ode.rhsextra
        self._dct       = ode.rhsdct 
        self.namespace  = model_data.namespace
        self.variables  = tuple(variables)
        self.functions  = tuple([ut.split_functionlike(f)[0] for f in functions])
        self.reactions  = tuple(reactions)
        self.extra      = () if extra is None else tuple(model_data['extra'].keys())
        
        #Assign data-related attributes
        self._state_dict     = model_data['states']
        self._parameter_dict = model_data['parameters']
        self.ref             = ref
        self.state_names     = tuple(model_data['states'].keys())
        self.parameter_names = tuple(model_data['parameters'].keys())
        self.states          = states
        self.parameters      = parameters
        self.int_args        = int_args
        self.sim_args        = {} if sim_args   is None else sim_args
        self.optim_args      = {} if optim_args is None else optim_args
        self.data_args       = {} if data_args  is None else data_args
        self.tspan           = tspan
        self.meta            = meta
        
    ###########################################################################
    #Attribute Management
    ###########################################################################
    @property
    def states(self) -> pd.DataFrame:
        return self._state_dict.df
    
    @states.setter
    def states(self, mapping: Union[dict, pd.DataFrame, pd.Series]) -> None:
        state_dict = self._state_dict
        df         = state_dict.mapping2df(mapping)
        
        self._state_dict.df = df
    
    @property
    def _states(self) -> dict:
        return self._state_dict.by_index()
    
    @property
    def parameters(self) -> pd.DataFrame:
        return self._parameter_dict.df
    
    @parameters.setter
    def parameters(self, mapping: Union[dict, pd.DataFrame, pd.Series]) -> None:
        parameter_dict = self._parameter_dict
        df         = parameter_dict.mapping2df(mapping)
        
        self._parameter_dict.df = df
    
    @property
    def _parameters(self) -> dict:
        return self._parameter_dict.by_index()
    
    def __setattr__(self, attr: str, value: Any) -> None:
        if attr in self._locked and hasattr(self, attr):
            raise AttributeError(f'{attr} attribute is locked.')
        
        elif attr == 'tspan':
            if value is None:
                super().__setattr__(attr, {})
                return
            elif not ut.isdictlike(value):
                raise TypeError('tspan must be a dict.')
            
            #Check dimensions and type
            for scenario, v in value.items():
                arr = np.array(v)
                if len(arr.shape) != 1:
                    raise ValueError('tspan for each scenario must be one dimensional.')
                
            super().__setattr__(attr, value)
            
        else:
            super().__setattr__(attr, value)
    
    def get_tspan(self, scenario: Scenario) -> np.ndarray:
        tspan = self.tspan 
        
        if self.tspan:
            return tspan.get(scenario, self.default_tspan)
        else:
            return self.default_tspan
            
    
    ###########################################################################
    #Integration
    ###########################################################################
    def __call__(self, 
                 scenario       = None, 
                 y0             = None, 
                 p0             = None, 
                 tspan          = None, 
                 overlap        = True, 
                 raw            = False, 
                 include_events = True,
                 **int_args
                 ) -> np.array:
        
        def _2array(vals, default, variables, name):
            if vals is None:
                return default[scenario]
            elif ut.islistlike(vals):
                arr = np.array(vals)
            elif type(vals) == pd.DataFrame:
                arr = vals.loc[scenario][variables].values
            elif type(vals) == pd.Series:
                arr = vals[variables].values
                
            elif ut.isdictlike(vals): 
                arr = np.array([default[scenario][v] for v in variables])
            else:
                msg = f'{name} must be an array, dict, DataFrame or Series. Received {type(vals)}'
                raise TypeError(msg)
            
            if len(arr.shape) == 1:
                return arr
            else:
                raise ValueError(f'{name} could not be formatted into a 1-D array.')
            
        if tspan is None:
            tspan = np.array(self.tspan.get(scenario, self.default_tspan))
        elif type(tspan) == dict:
            tspan = np.array(tspan[scenario])
        elif ut.islistlike(tspan):
            tspan = np.array(tspan)
        else:
            raise TypeError(f'tspan must be a dict or array. Received {type(tspan)}')
        
        if len(tspan.shape) != 1:
            raise ValueError('tspan must be 1-D.')
        
        #Reassign and/or extract
        y0 = _2array(y0, self._states, self.state_names, 'states')
        p0 = _2array(p0, self._parameters, self.parameter_names, 'parameters')

        tspan     = self.get_tspan(scenario) if tspan is None else tspan
        
        #Reassign and/or extract
        events   = self._events
        int_args = self.int_args if self.int_args else {}
        
        t, y, p = ivp.integrate(self._rhs, 
                                tspan          = tspan, 
                                y0             = y0, 
                                p0             = p0, 
                                events         = events, 
                                overlap        = overlap, 
                                include_events = include_events,
                                **int_args
                                )
            
        if raw:
            return t, y, p
        else:
            return ODEResult(t, y, p, self)
    
    def integrate(self, **kwargs):
        scenarios = list(self.states.index)
        result    = {}
        
        for c in scenarios:
            kwargs_   = {k: v.get(c) for k, v in kwargs.items()}
            result[c] = self(c, **kwargs_)
            
        return result
    
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
    def __init__(self, t: np.ndarray, y: np.ndarray, p: np.ndarray, model: ODEModel):
        state_names     = model.state_names
        parameter_names = model.parameter_names
        to_remove       = set(model.functions)
        namespace       = model.namespace.difference(to_remove)
        namespace       = namespace | {ut.diff(x) for x in model.state_names}
        
        self.t               = t
        self.y               = dict(zip(state_names, y))
        self.p               = dict(zip(parameter_names, p)) 
        self._eval_extra     = None
        self._eval_dct       = None 
        self.extra           = model._extra
        self.dct             = model._dct
        self._args           = [t, y, p]
        self.namespace       = frozenset(namespace)
        self.extra_variables = model.extra
        
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
    
    def __getitem__(self, var):
        return self.get(var)
    
    def __setitem__(self, var, vals):
        self.evaluated[var] = vals
    
    def get(self, var, default='__error__'):
        if ut.islistlike(var):
            return [self.get(v) for v in var]
        elif var == 'time':
            return self.t
        elif var in self.y:
            return self.y[var]
        elif var in self.p:
            return self.p[var]
        elif var in self.extra_variables:
            return self.eval_extra[var]    
        elif var in self.eval_dct:
            return self.eval_dct[var]
        elif default == '__error__':
            raise ValueError(f'{str(self)} does not contain "{var}"')
        else:
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
    
