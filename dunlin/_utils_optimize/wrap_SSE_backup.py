import numpy    as np
import warnings
from   numba    import njit

###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin.simulate as sim

###############################################################################
#Error Calculation
###############################################################################
class SSECalculator():
    ###########################################################################
    #SSE Calculation
    ###########################################################################       
    @staticmethod
    @njit
    def reconstruct(nominal, sampled_index, scaled_free_params_array):
        params                = nominal.copy()
        params[sampled_index] = scaled_free_params_array
        
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
        free_params                              = model.optim_args['free_params'] 
        tspan, t_data, y_data, s_data, exv_names = split_dataset(model, dataset)
        init                                     = get_init(model)
        state_index                              = dict(zip(model._states, range(len(model._states))))
        
        nominal_vals  = self.sort_params(model.params, model.states)
        nominal_vals  = dict(zip(nominal_vals.index, nominal_vals.values))
        param_names   = model.get_param_names()
        param_index   = {p: i for i, p in enumerate(param_names)} 
        sampled_index = [param_index[p] for p in free_params]
        
        #Check
        param_check = set(param_names) 
        if len(param_check.intersection(free_params)) != len(free_params):
            raise ValueError('Unexpected parameter(s) in free params.')
        
        #Assign
        self.tspan         = tspan
        self.t_data        = t_data
        self.y_data        = y_data
        self.s_data        = s_data
        self.exv_names     = exv_names
        self.init          = init
        self.state_index   = state_index
        self.model         = model
        self.nominal       = nominal_vals
        self.sampled_index = np.array(sampled_index)
        
        #For testing/development
        self.disp = False
        
    ###########################################################################
    #Integration
    ###########################################################################       
    def get_SSE(self, free_params_array):
        SSE         = 0
        tspan       = self.tspan
        t_data      = self.t_data
        y_data      = self.y_data
        s_data      = self.s_data
        exv_names   = self.exv_names
        init        = self.init
        state_index = self.state_index
        model       = self.model
        
        for scenario, y0 in init.items():
            if scenario not in tspan:
                continue

            params_array = self.reconstruct(self.nominal[scenario], 
                                            self.sampled_index, 
                                            free_params_array
                                        )
            if self.disp:
                print(scenario)
                print(params_array)
            t, y = model.integrate(scenario, 
                                   y0, 
                                   params_array, 
                                   overlap        = False, 
                                   include_events = False, 
                                   tspan          = tspan[scenario]
                                   )
            
            if exv_names:
                y = sim.IntResult(model, t, y, params_array, scenario, exv_names)

            for var, yd in t_data[scenario].items():
                ym   = y.get1d(var) if exv_names else y[state_index[var]]
                idx  = t_data[scenario][var]
                yd   = y_data[scenario][var]
                sd   = s_data[var]
                SSE += self.get_error(ym, yd, sd, idx) if hasattr(ym, '__iter__') else self.get_error(ym, yd, sd, None)
                
        return SSE
    
    def __call__(self, params_array):
        return self.get_SSE(params_array)
    
    ###########################################################################
    #Miscellaneous
    ###########################################################################       
    def __repr__(self):
        return f'{type(self).__name__}<model: {self.model.model_key}>'
    
    def __str__(self):
        return self.__repr__()
        
###############################################################################
#Preprocessing
###############################################################################
def check_scenarios_match(model, dataset):
    d_scenarios = set([k[1] for k in dataset.keys()])
    m_scenarios = set(model.states.index)
    intersect   = d_scenarios.intersection(m_scenarios)
    if len(intersect) != len(d_scenarios):
        msg = f'There are one or scenarios in dataset that are not in model.states for model {model.model_key}'
        warnings.warn(msg)
    
    
def split_dataset(model, dataset):    
    states     = model.get_state_names()
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
    exv_names  = []
    
    for key, value in dataset.items():
        if key[0] == 'Data':
            _, scenario, var = key
            
            #Update y_data
            y_data.setdefault(scenario, {})[var] = np.array(value)
            
            #Check if variable is a state or exv
            if var in states:
                y_set.add((scenario, var))
            elif var in model_exvs:
                exv_names.append(var)
            else:
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
    
    return tspan, t_data, y_data, s_data, exv_names

def get_init(model):
    init = {k: v.values for k, v in model.states.T.to_dict('series').items()}
    return init

def get_nominal(model, free_params, nominal=None):
    if nominal == None:
        if len(model.states) != len(model.params):
            raise ValueError('model.states and model.params do not have the same length.')
        nominal_vals = dict(zip(model.states.index, model.params.values))
    else:
        temp         = model.loc[nominal]
        nominal_vals = {i: temp for i in model.states.index}
    return nominal_vals
