import matplotlib.axes as axes
import numpy           as np
import pandas          as pd
import warnings
from matplotlib.collections import LineCollection
from numba                  import njit
from numbers                import Number
from typing                 import Literal

import dunlin.utils      as ut
import dunlin.utils_plot as upp
from ..ode.odemodel import ODEModel, is_scenario

Parameter = str
State     = str
Scenario  = Number|str

class SSECalculator:
    @staticmethod
    def parse_data(model : ODEModel,
                   data  : dict[Scenario, dict[State, pd.Series]]|dict[State, dict[Scenario, pd.Series]], 
                   by    : Literal['scenario', 'state'] = 'scenario'
                   ):
        #Check input
        if by != 'scenario' and by!= 'state':
            msg = f'The "by" argument must be "scenario" or "state". Received {by}.'
            raise ValueError(msg)
        
        #Extract the data and its corresponding time points
        #Calculate or extract standard deviation 
        #Each dict maps scenario -> state -> array of data/time points
        scenario2y_data  = {}
        scenario2t_data  = {}
        scenario2sd_data = {} 
        
        #Determine the time points for numerical integration
        #The dict maps scenario -> set of time points
        scenario2tpoints = {}
        
        #SSE can be calculated for the following
        allowed = {*model.states, 
                   *model.variables, 
                   *getattr(model, 'extra', [])
                   }
        
        #Make cleaned copies of the data
        cleaned_data = {}
        
        for first, dct in data.items():
            for second, series in dct.items():
                #Clean the series
                series = series.dropna()
                
                #Determine the scenario and state
                if by == 'scenario':
                    scenario = first
                    state    = second
                else:
                    scenario = second
                    state    = first
                
                #Check the scenarios and state
                if state not in allowed:
                    msg = f'Data contains the state "{state}" not found in model {model.ref}.'
                    raise ValueError(msg)
                
                
                if scenario not in model.state_dict:
                    msg = f'Data contains a scenario "{scenario}" not found in model {model.ref}.'
                    raise ValueError(msg)
                
                #Reformat the series if it has a multiindex
                default  = np.percentile(series, 75)/20
                if series.index.nlevels == 1:
                    temp     = pd.Series(np.zeros(len(series)), index=series.index)
                    sd_      = getattr(series, 'sd', temp)
                    sd       = getattr(series, 'sd', default)
                    y_array  = series.values
                    t_array  = series.index.values
                    
                else:
                    if 'time' in series.index.names:
                        groupby = series.groupby(by='time')
                    else:
                        groupby = series.groupby(level=0)
                    
                    mean    = groupby.mean()
                    sd_     = groupby.std()
                    sd     = sd.fillna(default)
                    y_array = mean
                    t_array = series.index.values
                    
                #Update the result
                scenario2y_data.setdefault(scenario, {})
                scenario2y_data[scenario][state] = y_array
                
                scenario2t_data.setdefault(scenario, {})
                scenario2t_data[scenario][state] = t_array
                
                scenario2sd_data.setdefault(scenario, {})
                scenario2sd_data[scenario][state] = sd
                
                scenario2tpoints.setdefault(scenario, set())
                scenario2tpoints[scenario].update(t_array)
                
                cleaned_data.setdefault(scenario, {})
                series.sd_                    = sd_  
                cleaned_data[scenario][state] = series
                
        #Determine the indices for extracting y_model
        #Make the tspans for numerical integration
        scenario2t_idxs = {}
        scenario2tspan  = {}
        
        for scenario, tpoints in scenario2tpoints.items():
            #Make the tspan for numerical integration
            #Update the results
            tspan = np.array(sorted(tpoints))
            scenario2tspan[scenario] = tspan
            
            #Determine the indices for extracting y_model 
            tpoint2idx = {tpoint: i for i, tpoint in enumerate(tspan)}
            
            state2t_idxs = scenario2t_idxs.setdefault(scenario, {})
            for state, tpoints_ in scenario2t_data[scenario].items():
                t_idxs              = np.array([tpoint2idx[i] for i in tpoints_])
                state2t_idxs[state] = t_idxs
            
        return (scenario2y_data, 
                scenario2t_data, 
                scenario2sd_data, 
                scenario2t_idxs, 
                scenario2tspan,
                cleaned_data
                )
    
    ###########################################################################
    #SSE Calculation
    ###########################################################################       
    def reconstruct(self, free_params_array):
        sampled_index = self.sampled_parameter_idxs
        nominal_dct   = self.nominal
        p             = {} 
        
        for scenario in self.scenario2y0:
            recon_array = self._reconstruct(nominal_dct[scenario], sampled_index, free_params_array)
            p[scenario] = recon_array
        
        return p
        
    @staticmethod
    @njit
    def _reconstruct(nominal, sampled_index, free_params_array):
        params                = nominal.copy()
        params[sampled_index] = free_params_array
        
        return params
    
    @staticmethod
    @njit
    def get_error(ym, yd, sd, idx):
        ym_ = ym if idx is None else ym[idx]
        # return np.sum(np.abs(ym_-yd)**2/sd**2)
        return np.sum((ym_-yd)**2/sd)
    
    @staticmethod
    def sort_params(params, states):
        try:
            return params.loc[states.index]
        except:
            raise ValueError('Parameters are missing one or more indices.')
            
    ###########################################################################
    #Instantiators
    ###########################################################################       
    def __init__(self, 
                 model : ODEModel,
                 data  : dict[Scenario, dict[State, pd.Series]]|dict[State, dict[Scenario, pd.Series]], 
                 by    : Literal['scenario', 'state'] = 'scenario'
                 ):
        
        #Determine the free parameters
        free_parameters = model.opt_args['free_parameters'] 
        scenario2y0     = model.state_dict
        
        nominal                = model.parameter_dict
        parameters             = model.parameters
        sampled_parameter_idxs = [i for i, p in enumerate(parameters) if p in free_parameters]
        
        #Check the validity of the free parameters
        allowed    = set(parameters)
        received   = set(free_parameters)
        unexpected = received - allowed
        if unexpected:
            raise ValueError('Unexpected parameter(s) in free parameters: {unexpected}.')
        
        #Assign attributes
        self.ref                    = model.ref
        self.scenario2y0            = scenario2y0 
        self.nominal                = nominal
        self.sampled_parameter_idxs = np.array(sampled_parameter_idxs)
        self.model                  = model
        
        if model.data_args: 
            self.data_args = model.data_args
        else:
            self.data_args = getattr(model, 'sim_args', {}).get('line_args', {})
        
        #Preprocess and get mappings
        (self.scenario2y_data, 
         self.scenario2t_data, 
         self.scenario2sd_data, 
         self.scenario2t_idxs, 
         self.scenario2tspan,
         self.data
         ) = self.parse_data(model, data)
        
    ###########################################################################
    #Integration
    ###########################################################################       
    def __call__(self, free_params_array, combine=True):
        model            = self.model
        scenarios        = model.state_dict.keys()
        SSE              = 0 if combine else dict.fromkeys(scenarios, 0)
        scenario2y0      = self.scenario2y0
        scenario2p0      = self.reconstruct(free_params_array)
        scenario2y_data  = self.scenario2y_data
        scenario2t_idxs  = self.scenario2t_idxs
        scenario2sd_data = self.scenario2sd_data
        scenario2tspan   = self.scenario2tspan
        
        for scenario in scenarios:
            if scenario not in scenario2y_data:
                continue
            
            ir = model(y0             = scenario2y0[scenario],
                       p0             = scenario2p0[scenario], 
                       tspan          = scenario2tspan[scenario], 
                       include_events = False
                       )
            
            #Note that state can actually be a variable
            for state, y_data in scenario2y_data[scenario].items():
                y_model  = ir[state]
                idx      = scenario2t_idxs[scenario][state]
                sd       = scenario2sd_data[scenario][state]
                
                SSE_ = self.get_error(y_model, y_data, sd, idx) 
                
                if combine:
                    SSE += SSE_
                else:
                    SSE[scenario] += SSE_
                    
        return SSE
    
    ###########################################################################
    #Representation
    ###########################################################################       
    def __repr__(self):
        return f'{type(self).__name__}({self.ref})'
    
    def __str__(self):
        return self.__repr__()
    
    ###########################################################################
    #Plotting
    ###########################################################################
    def plot_data(self,
                  ax            : axes.Axes,
                  var           : str|tuple[str, str],
                  scenarios     : Scenario|list[Scenario] = None,
                  **kwargs
                  ) -> axes.Axes:
        match scenarios:
            case None:
                scenarios = set(self.data)
            case [*scenarios]:
                scenarios = set(scenarios)
            case c if isinstance(c, Scenario):
                scenarios = {c}
            case _:
                msg = ''
                raise ValueError(msg)
        
        result = {}
        for scenario in scenarios:
            result[scenario] = self._plot_data(ax, var, scenario, **kwargs)
            
        return result
    
    def _plot_data(self,
                   ax       : axes.Axes,
                   var      : str|tuple[str, str],
                   scenario : Scenario,
                   **kwargs
                   ) -> axes.Axes:
        
        match var:
            case str(y):
                x_vals = self.data[scenario][y].index
                y_vals = self.data[scenario][y].values
                
            case [str(x), str(y)]:
                x_series = self.data[scenario][x]
                y_series = self.data[scenario][y]
                
                #The two series may not have the same indices
                #Get only the common indices
                common = x_series.index.intersection(y_series.index)
                x_vals = x_series.loc[common].values
                y_vals = y_series.loc[common].values
            
            case _:
                msg = f'Could not parse the var argument {var}.'
                raise ValueError(msg)
        
        label      = lambda ref, scenario, var: '{} {} {}'.format(ref, scenario, var)
        default    = {'linestyle'  : 'None', 
                      'marker'     : 'o', 
                      'markersize' : 5, 
                      'label'      : label,
                      **self.data_args
                      }
        sub_args   = {'ref': self.ref, 'scenario': scenario, 'var': var}
        converters = {'color'  : upp.get_color,
                      'colors' : upp.get_colors
                      }
        kwargs     = upp.process_kwargs(kwargs, 
                                        [scenario, var], 
                                        default    = {**default, 'label': label},
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
    
        