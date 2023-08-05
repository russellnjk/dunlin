import numpy    as np
import pandas   as pd
import warnings
from numba   import njit
from numbers import Number
from typing  import Literal

import dunlin.utils as ut
from ..ode.odemodel import ODEModel

Parameter = str
State     = str
Scenario  = Number|str|tuple[Number|str]

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
        
        for first, dct in data.items():
            for second, series in dct.items():
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
                if series.index.nlevels == 1:
                    default  = np.percentile(series, 75)/20
                    sd       = getattr(series, 'sd', default)
                    y_array  = series.values
                    t_array  = series.index.values
                    
                else:
                    msg = ''
                    raise NotImplementedError(msg)
                
                #Update the result
                scenario2y_data.setdefault(scenario, {})
                scenario2y_data[scenario][state] = y_array
                
                scenario2t_data.setdefault(scenario, {})
                scenario2t_data[scenario][state] = t_array
                
                scenario2sd_data.setdefault(scenario, {})
                scenario2sd_data[scenario][state] = sd
                
                scenario2tpoints.setdefault(scenario, set())
                scenario2tpoints[scenario].update(t_array)
                

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
                scenario2tspan
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
        free_parameters = model.optim_args['free_parameters'] 
        scenario2y0     = model.state_dict
        
        nominal               = model.parameter_dict
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
        self.nominal               = nominal
        self.sampled_parameter_idxs = np.array(sampled_parameter_idxs)
        self.model                  = model
        
        #Preprocess and get mappings
        (self.scenario2y_data, 
         self.scenario2t_data, 
         self.scenario2sd_data, 
         self.scenario2t_idxs, 
         self.scenario2tspan
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
    
        