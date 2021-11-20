import numpy    as np
import warnings
from   numba    import njit

class SSECalculator():
    ###########################################################################
    #SSE Calculation
    ###########################################################################       
    def reconstruct(self, free_params_array):
        sampled_index = self.sampled_index
        nominal_dct   = self.nominal
        p             = {} 
        
        for scenario in self.init:
            nominal     = self.nominal[scenario]
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
        return np.sum(np.abs(ym_-yd)**2)/sd
    
    @staticmethod
    def sort_params(params, states):
        try:
            return params.loc[states.index]
        except:
            raise ValueError('Parameters are missing one or more indices.')
            
    ###########################################################################
    #Instantiators
    ###########################################################################       
    def __init__(self, model, dataset):
        free_params                   = model.optim_args['free_params'] 
        tspan, t_data, y_data, s_data = split_dataset(model, dataset)
        init                          = model._states
        
        nominal_vals  = model._params
        param_names   = model.get_param_names()
        sampled_index = [i for i, p in enumerate(param_names) if p in free_params]
        
        #Check
        param_check = set(param_names) 
        if len(param_check.intersection(free_params)) != len(free_params):
            raise ValueError('Unexpected parameter(s) in free params.')
        
        #Assign
        self.tspan         = tspan
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
    def __call__(self, free_params_array):
        SSE     = 0
        t_data  = self.t_data
        y_data  = self.y_data
        s_data  = self.s_data
        model   = self.model
        p       = self.reconstruct(free_params_array)
        
        for ir in model(p=p, tspan=self.tspan, overlap=False, include_events=False):
            scenario = ir.scenario
            
            for var, yd in t_data.get(scenario, {}).items():
                ym  = ir[var]
                idx = t_data[scenario][var]
                yd  = y_data[scenario][var]
                sd  = s_data[var]
                
                if type(ym) == dict:
                    ym = ym['y']
                
                SSE += self.get_error(ym, yd, sd, idx) if hasattr(ym, '__iter__') else self.get_error(ym, yd, sd, None) 
           
        return SSE
    
    ###########################################################################
    #Miscellaneous
    ###########################################################################       
    def __repr__(self):
        return f'{type(self).__name__}(model: {self.model.model_key})'
    
    def __str__(self):
        return self.__repr__()
        
###############################################################################
#Preprocessing
###############################################################################
def check_scenarios_match(model, dataset):
    
    d_scenarios = set([k[1] for k in dataset.keys()])
    m_scenarios = set(model.states.index)
    diff1       = d_scenarios.difference(m_scenarios)
    diff2       = m_scenarios.difference(d_scenarios)
    
    if diff1:
        msg = f'There are one or scenarios in dataset that are not in model.states for model {model.model_key}'
        warnings.warn(msg)
    
    
    if diff2:
        msg = f'There are one or scenarios in dataset that are missing from model.states for model {model.model_key}'
        warnings.warn(msg)
    
def split_dataset(model, dataset):    
    mod_vars  = [*model.get_state_names(), *model.vrbs] if model.vrbs else model.get_state_names()
    model_exvs = model.exvs if model.exvs else []
    y_data     = {}
    t_data     = {}
    s_data     = {} 
    y_set      = set()
    x_set      = set()
    t_set      = set()
    s_set      = set()
    t_len      = {}
    tpoints    = {}
    max_vals   = {}
    
    for key, value in dataset.items():
        if key[0] == 'Data':
            _, scenario, var = key
            
            #Update y_data
            y_data.setdefault(scenario, {})[var] = np.array(value)
            
            #Check if variable is a state or exv
            if var in mod_vars :
                y_set.add((scenario, var))
            elif var not in model_exvs:
                raise ValueError(f'exp_data contains an exv {var} which is not in the model.')
            
            #Track variable to determine if sd is provided
            x_set.add(var)
            
            #Track max value for sd calculations
            max_vals.setdefault(var, [])
            max_vals[var] += [*value]
            
        elif key[0] == 'Time':
            _, scenario, var = key
            
            #Create a list to store time points 
            tpoints.setdefault(scenario, [])
            
            #Update t_data
            #Check length of tpoints[scenario]
            #Determine the start and stop indices
            t_data.setdefault(scenario, {}) 
            t_data[scenario][var] = len(tpoints[scenario]), len(tpoints[scenario]) + len(value)
            
            #Track to see if the corresponding y_data is availabe
            t_set.add((scenario, var))
            
            #Update tpoints AFTER update t_data
            tpoints[scenario] += [*value]
            t_len[(scenario, var)] = len(value)
        
        elif key[0] == 'Std':
            _, scenario, var = key
            
            #Update s_data
            s_data[var] = max(value, s_data.get(value, 0))
            
            #Track variables for which the user has provided sd
            s_set.add(var)

    #Check that all states have an associated time
    #Does not apply to exvs
    if len(y_set.intersection(t_set)) != len(y_set):
        raise ValueError()
    
    #Check that t and y arrays have the same length
    mismatched = [k for k, v in t_len.items() if len(y_data[k[0]][k[1]]) != v]
    if mismatched:
        raise ValueError('Mismatched data lengths.')
    
    #Make sd data if absent
    for state in x_set.difference(s_set):
        s_data[state] = np.percentile(max_vals[state], 75)/20
    
    #Make tspan for SSE calculation
    #Get the indices of tspan for each state/scenario
    tspan   = {}
    for scenario, tps in tpoints.items():
        tspan_, indices = np.unique(tps, return_inverse=True)
        if not len(tspan_):
            raise ValueError('tspan cannot have only 1 time point.')
        
        if tspan_[0] <= 0:
            delta           = 0
            tspan[scenario] = tspan_
        else:
            delta           = 1
            tspan[scenario] = np.array([0, *tspan_])
        
        for var in t_data[scenario]:
            start, stop = t_data[scenario][var]
            indices_    = indices[start:stop] + delta
            
            #Reassign with the indices
            t_data[scenario][var] = indices_
    
    #Check that scenarios match
    check_scenarios_match(model, dataset)
    
    return tspan, t_data, y_data, s_data
