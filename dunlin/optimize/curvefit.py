import matplotlib.axes as axes
import numpy           as np
import pandas          as pd
from matplotlib.collections import LineCollection
from typing                 import Callable, Literal

###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin.optimize.optimizer as opt
from ..ode.odemodel import ODEModel, ODEResultDict
from .wrap_SSE      import SSECalculator, State, Parameter, Scenario

###############################################################################
#Curvefitter
###############################################################################
class Curvefitter(opt.Optimizer):
    def __init__(self, 
                 model            : ODEModel,
                 data             : dict[tuple[State, Scenario], pd.Series],
                 _nominal : pd.DataFrame = None
                 ):
        
        get_SSE    = SSECalculator(model, data)
        
        #Instantiate
        nominal         = model.parameter_df if _nominal is None else _nominal
        free_parameters = model.opt_args.get('free_parameters', {}) 
        opt_args        = model.opt_args.get('opt_args',    {})
        trace_args      = model.trace_args
        
        super().__init__(nominal, 
                         free_parameters, 
                         get_SSE, 
                         opt_args, 
                         trace_args
                         )

        #Allows access to the model's methods later
        #Add the model as an attribute
        self.ref   = model.ref
        self.model = model
        self.data  = get_SSE.data
        self._data = {k: v.copy() for k, v in data.items()}
        
        #Create a cache for the integration results
        self.cache = {}
    
    ###########################################################################
    #Access and Modification
    ###########################################################################
    @property
    def sse_calc(self) -> Callable:
        return self.neg_log_likelihood
    
    def get(self, 
            n           : int|list[int] = 0, 
            sort        : bool          = True, 
            _df         : bool          = True
            ) -> dict|dict[int, dict]:
        if type(n) == list:
            return {i: self.get(i, sort) for i in n}
        elif type(n) != int:
            msg = f'Argument n must be an integer. Received {type(n)}.'
            raise TypeError(msg)
        
        #Use the get method of the trace
        result = self.trace.get(n, sort)
        row    = result['sample']
        dct    = self.sse_calc.reconstruct(row.values)
        
        result['parameter_dict'] = dct
        
        if _df:
            df         = pd.DataFrame.from_dict(result['parameter_dict'], 
                                                orient='index'
                                                )
            df.columns = self.model.parameters                     
            
            #Update the results
            result['parameter_df'] = df
            
        return result
    
    ###############################################################################
    #Seeding
    ###############################################################################       
    def seed(self, new_name=0):
        nominal = self.nominal.copy()
        for sp in self.sampled_parameters:
            new_val          = sp.new_sample()
            nominal[sp.name] = new_val

        return type(self)(self.model, self._data, nominal)    
    
    ###########################################################################
    #Simulation
    ###########################################################################
    def integrate(self, 
                  n              : int|list[int]  = 0, 
                  sort           : bool           = True, 
                  scenarios      : list[Scenario] = None,
                  raw            : bool           = False,
                  include_events : bool           = True,
                  ) -> ODEResultDict:
        if type(n) == list:
            return [self.simulate(i, sort) for i in n]
        elif type(n) != int:
            msg = f'Argument n must be an integer. Received {type(n)}.'
            raise TypeError(msg)
            
        search_result = self.get(n, sort, _df=False)
        idx           = search_result['sample'].name
        
        if idx in self.cache:
            resultdict     = self.cache[idx]
            all_scenarios  = set(self.sse_calc.nominal)
            scenarios      = all_scenarios if scenarios is None else scenarios
            to_integrate   = all_scenarios - scenarios
            
            if to_integrate:
                parameter_dict  = search_result['parameter_dict']
                temp            = self.model.integrate(scenarios = to_integrate, 
                                                       _p0       = parameter_dict
                                                       )
                to_update       = temp.scenario2intresult
                
                resultdict.scenario2intresult.update(to_update)
            
            return resultdict
        else:
            parameter_dict  = search_result['parameter_dict']
            resultdict      = self.model.integrate(scenarios = scenarios, 
                                                   _p0       = parameter_dict
                                                   )
            self.cache[idx] = resultdict
            
            return resultdict
    
    def plot_line(self, 
                  ax             : axes.Axes,
                  var            : str|tuple[str, str],
                  n              : int|list[int]  = 0, 
                  sort           : bool           = True, 
                  scenario      : list[Scenario]  = None,
                  **kwargs
                  ) -> axes.Axes:
        
        resultdict = self.integrate(n, sort, scenario)
        return resultdict.plot_line(ax, var, **kwargs)
    
    def plot_data(self,
                  ax            : axes.Axes,
                  var           : str|tuple[str, str],
                  scenarios     : Scenario|list[Scenario] = None,
                  **kwargs
                  ) -> axes.Axes:
        return self.sse_calc.plot_data(ax, 
                                       var       = var, 
                                       scenarios = scenarios, 
                                       **kwargs
                                       )
        
        
    def plot_result(self,
                    ax           : axes.Axes,
                    var          : str|tuple[str, str],
                    scenarios    : list[Scenario] = None,
                    guess_ax     : axes.Axes      = None,
                    data_ax      : axes.Axes      = None,
                    guess_kwargs : dict           = None,
                    data_kwargs  : dict           = None,
                    **kwargs
                    ) -> axes.Axes:
        result = []
        
        #Plot the data
        if self.sse_calc.contains_var(var):
            data_ax     = ax if data_ax     is None else ax 
            data_kwargs = {} if data_kwargs is None else data_kwargs
            
            r = self.plot_data(data_ax, 
                               var       = var, 
                               scenarios = scenarios, 
                               **data_kwargs
                               )
            result.append(r)
        
        #Plot the guess first
        default      = {'linestyle' : '--'}
        guess_kwargs = default if guess_kwargs is None else {**default, **guess_kwargs}
        guess_ax     = ax      if guess_ax     is None else guess_ax
        
        r = self.plot_line(guess_ax, 
                           var       = var, 
                           n         = 0, 
                           sort      = False, 
                           scenarios = scenarios, 
                           **guess_kwargs
                           )
        result.append(r)
        
        #Plot the best 
        r = self.plot_line(ax, 
                           var       = var, 
                           n         = 0, 
                           sort      = True, 
                           scenarios = scenarios, 
                           **kwargs
                           )
        result.append(r)
        
        return result

###############################################################################
#High-Level Algorithms
###############################################################################
def fit_model(model : ODEModel, 
              data  : dict[tuple[State, Scenario], pd.Series], 
              runs  : int  = 1, 
              algo  : str  = 'differential_evolution', 
              lazy  : bool = False,
              **kwargs
              ) -> list[Curvefitter]:
    
    if lazy:
        return _fit_model_lazy(model, data, runs, algo, **kwargs)
    else:
        return _fit_model(model, data, runs, algo, **kwargs)

def _fit_model(model : ODEModel, 
               data  : dict[tuple[State, Scenario], pd.Series], 
               runs  : int  = 1, 
               algo  : str  = 'differential_evolution', 
               **kwargs
               ) -> list[Curvefitter]:
    curvefitters = []
    curvefitter  = None
    
    for i in range(runs):
        if i == 0:
            curvefitter = Curvefitter(model, data)
        else:
            curvefitter = curvefitter.seed(i)
        
        method = getattr(curvefitter, 'run_' + algo, None)
        if method is None:
            raise ValueError(f'No algorithm called "{algo}".')
            
        method(**kwargs)
        
        curvefitters.append(curvefitter)
        
    return curvefitters 

def _fit_model_lazy(model : ODEModel, 
                    data  : dict[tuple[State, Scenario], pd.Series], 
                    runs  : int  = 1, 
                    algo  : str  = 'differential_evolution', 
                    **kwargs
                    ) -> list[Curvefitter]:
    curvefitters = []
    curvefitter  = None
    
    for i in range(runs):
        if i == 0:
            curvefitter = Curvefitter(model, data)
        else:
            curvefitter = curvefitter.seed(i)
        
        method = getattr(curvefitter, 'run_' + algo, None)
        if method is None:
            raise ValueError(f'No algorithm called "{algo}".')
            
        method(**kwargs)
        
        curvefitters.append(curvefitter)
        
        yield curvefitter

