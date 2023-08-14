import matplotlib.axes as axes
import numpy           as np
import pandas          as pd
from matplotlib.collections import LineCollection
from typing                 import Callable, Literal

###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin.simulate           as sim
import dunlin.utils              as ut 
import dunlin.utils_plot         as upp 
import dunlin.optimize.optimizer as opt
import dunlin.data               as ddt
from ..ode.odemodel import ODEModel, ODEResultDict
from .wrap_SSE      import SSECalculator, State, Parameter, Scenario

###############################################################################
#High-Level Algorithms
###############################################################################
def fit_model(model : ODEModel, 
              data  : dict[Scenario, dict[State, pd.Series]]|dict[State, dict[Scenario, pd.Series]], 
              by    : Literal['scenario', 'state'] = 'scenario',
              runs  : int                          = 1, 
              algo  : str                          = 'differential_evolution', 
              **kwargs
              ):
    curvefitters = []
    curvefitter  = Curvefitter(model, data, by=by)
    
    for i in range(runs):
        if i > 0:
            curvefitter = curvefitter.seed(i)
        
        method = getattr(curvefitter, 'run_' + algo, None)
        if method is None:
            raise ValueError(f'No algorithm called "{algo}".')
            
        method(**kwargs)
        
        curvefitters.append(curvefitter)
        
    return curvefitters 

###############################################################################
#Curvefitter
###############################################################################
class Curvefitter(opt.Optimizer):
    def __init__(self, 
                 model : ODEModel,
                 data  : dict[Scenario, dict[State, pd.Series]]|dict[State, dict[Scenario, pd.Series]], 
                 by    : Literal['scenario', 'state'] = 'scenario',
                 ):
        
        get_SSE    = SSECalculator(model, data, by)
        
        #Instantiate
        nominal         = model.parameter_df
        free_parameters = model.optim_args.get('free_parameters', {})
        settings        = model.optim_args.get('settings',    {})
        trace_args      = model.trace_args
        
        super().__init__(nominal, 
                         free_parameters, 
                         get_SSE, 
                         settings, 
                         trace_args
                         )

        #Allows access to the model's methods later
        #Add the model as an attribute
        self.ref   = model.ref
        self.model = model
        self.data  = get_SSE.data
        self.by    = by
        
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
        