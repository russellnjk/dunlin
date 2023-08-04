import numpy    as np
import pandas   as pd
import warnings
from   numba    import njit

import dunlin.utils as ut
import dunlin.data  as ddt

class SSECalculator():
    
    @staticmethod
    def parse_df(model, *dfs, const_sd=False):
        #Create Dataset
        if not dfs:
            raise ValueError('No data provided.')
        #Case 1: A Dataset is provided
        elif len(dfs) == 1 and type(dfs[0]) == ddt.TimeResponseData:
            dataset = dfs[0]
            no_fit  = set(dataset.no_fit)
            dataset = dataset.data
        #Case 2: 
        elif all([type(df) == pd.DataFrame for df in dfs]):
            #Create a dataset in dict format
            dataset, data_namespace, data_scenarios = ddt.TimeResponseData.split_df(*dfs)
            no_fit                                  = set()        
        else:
            msg = 'Expected a single TimeResponseData or an iterable of DataFrames.'
            raise TypeError(msg)
        
        #Prepare to extract the information
        mod_vars   = [*model.state_names, *model.variables] 
        mod_extras = list(model.extra) if model.extra else []
        y_data     = {}
        t_data     = {}
        s_data     = {} 
        t_idx      = {}
        tpoints    = {}
        tspan      = {}
        ypoints    = {}
        
        #Parse y_data
        for scenario, c_data in dataset.items():
            for variable, series in c_data.items():
                try:
                    ut.check_valid_name(variable)
                except:
                    continue
                
                if variable in no_fit:
                    continue
                
                elif variable in mod_vars:
                    
                    series   = series.dropna().astype(np.float64)
                    
                    sd_var = '__' + variable
                    if sd_var in dataset.get(scenario, {}):
                        default_sd = dataset[scenario][sd_var] 
                        if default_sd.index.nlevels != 1:
                            raise ValueError('Series for std. dev. cannot have multiindex.')
                        elif not default_sd.index.is_unique:
                            raise ValueError('Series for std. dev. must have unique index.')
                    else:
                        default_sd = np.percentile(series, 75)/20
                    
                    gb = series.groupby(level=0)
                    y = gb.mean().values
                    t = np.array(list(gb.groups.keys()))
                    s = gb.std().fillna(default_sd).values
                    y_data.setdefault(scenario, {})[variable] = y
                    t_data.setdefault(scenario, {})[variable] = t
                    s_data.setdefault(scenario, {})[variable] = s
                    
                    #Use for default sd
                    ypoints.setdefault(variable, set()).update(y)
                    
                    #Ensure 0 is inside to comply with ivp function requirements
                    tpoints.setdefault(scenario, [0]).extend(t)
                    
                elif variable in mod_extras:
                    if ut.isnum(series):
                        y_data.setdefault(scenario, {})[variable] = np.float64(series)
                        s_data.setdefault(scenario, {})[variable] = 1
                    else:
                        raise TypeError('Values for extra variables must be a number.')
                else:
                    msg = f'Variable {variable} is not in  model.'
                    raise ValueError(msg)
                    
        #Should not be required any more
        # print(s_data)
        # assert False
        #Fill missing sd or replace with default
        # for variable in ypoints:
        #     ypoints[variable] = np.percentile(list(ypoints[variable]), 75)/20
            
        # for scenario, dct in s_data.items():
        #     for variable in dct:
        #         default = ypoints[variable]
        #         print(scenario, variable, default)
        #         if const_sd:
        #             dct[variable] = default
        #         else:
        #             dct[variable] = dct[variable].fillna(default).values
                
        #Determine the tspan and indices
        for scenario, values in tpoints.items():
            tspan_ = np.unique(list(values))
            if not len(tspan_):
                raise ValueError('tspan cannot have only 1 time point.')
                
            tspan[scenario] = tspan_
            idxs            = {k: v for v, k in enumerate(tspan_)}
            t_idx.setdefault(scenario, {})
            for variable, t_variable in t_data[scenario].items():
                
                if variable in mod_vars:
                    idx = np.array([idxs[tp] for tp in t_variable])
                    
                    t_idx[scenario][variable] = idx
                else:
                    t_idx[scenario][variable] = None

        return tspan, y_data, s_data, t_data, t_idx
    
    ###########################################################################
    #SSE Calculation
    ###########################################################################       
    def reconstruct(self, free_params_array):
        sampled_index = self.sampled_index
        nominal_dct   = self.nominal
        p             = {} 
        
        for scenario in self.init:
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
    def __init__(self, model, *dfs, const_sd=False):
        tspan, y_data, s_data, t_data, t_idx = self.parse_df(model, *dfs, const_sd=const_sd)

        free_params  = model.optim_args['free_parameters'] 
        init         = model._states
        
        nominal_vals  = model._parameters
        param_names   = model.parameter_names
        sampled_index = [i for i, p in enumerate(param_names) if p in free_params]
        
        #Check
        param_check = set(param_names) 
        if len(param_check.intersection(free_params)) != len(free_params):
            raise ValueError('Unexpected parameter(s) in free params.')
        
        #Assign
        self.tspan         = tspan
        self.t_idx         = t_idx
        self.t_data        = t_data
        self.y_data        = y_data
        self.s_data        = s_data
        self.init          = init
        self.model         = model
        self.nominal       = nominal_vals
        self.sampled_index = np.array(sampled_index)
        
    ###########################################################################
    #Integration
    ###########################################################################       
    def __call__(self, free_params_array, combine=True):
        model     = self.model
        scenarios = model.states.index
        SSE       = 0 if combine else dict.fromkeys(scenarios, 0)
        t_idx     = self.t_idx
        y_data    = self.y_data
        s_data    = self.s_data
        p         = self.reconstruct(free_params_array)
        
        for scenario in scenarios:
            ir = model(scenario=scenario,
                       y0=self.init[scenario],
                       p0=p[scenario], 
                       tspan=self.tspan, 
                       overlap=False, 
                       include_events=False
                       )
        
            for variable, yd in y_data.get(scenario, {}).items():
                ym  = ir[variable]
                idx = t_idx[scenario][variable]
                yd  = y_data[scenario][variable]
                sd  = s_data[scenario][variable]
                # print(yd[-5:])
                # print(ym[idx][-5:])
                # print(variable, scenario, self.get_error(ym, yd, sd, idx) )
                
                if combine:
                    SSE += self.get_error(ym, yd, sd, idx) 
                else:
                    SSE[scenario] += self.get_error(ym, yd, sd, idx) 
        return SSE
    
    ###########################################################################
    #Representation
    ###########################################################################       
    def __repr__(self):
        return f'{type(self).__name__}({self.model.ref})'
    
    def __str__(self):
        return self.__repr__()
    
        