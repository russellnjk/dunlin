import numpy   as     np
from   numba   import njit

###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin.simulation as sim

###############################################################################
#Error Calculation
###############################################################################
@njit
def get_error(ym, yd, sd, idx):
    ym_ = ym if idx is None else ym[idx]
    return np.sum(np.abs(ym_-yd)**2)/sd
    
def wrap_get_SSE(model, dataset):
    sse_calc = SSECalculator(model, dataset)
    return sse_calc.get_SSE
    # tspan, t_data, y_data, s_data, exv_names = split_dataset(model, dataset)
    # init                                     = get_init(model)
    # state_index                              = dict(zip(model._states, range(len(model._states))))
    
    # def get_SSE(params_array):
    #     SSE = 0
    #     for scenario, y0 in init.items():
    #         if scenario not in tspan:
    #             continue
            
    #         t, y = model.integrate(scenario, 
    #                                y0, 
    #                                params_array, 
    #                                overlap        = False, 
    #                                include_events = False, 
    #                                tspan          = tspan[scenario]
    #                                )
            
    #         if exv_names:
    #             y = sim.IntResult(model, t, y, params_array, scenario, exv_names)

    #         for var, yd in t_data[scenario].items():
    #             ym   = y.get1d(var) if exv_names else y[state_index[var]]
    #             idx  = t_data[scenario][var]
    #             yd   = y_data[scenario][var] 
    #             sd   = s_data[var]

    #             SSE += get_error(ym, yd, sd, idx) if hasattr(ym, '__iter__') else get_error(ym, yd, sd, None)

    #     return SSE
    # return get_SSE

class SSECalculator():
    def __init__(self, model, dataset):
        tspan, t_data, y_data, s_data, exv_names = split_dataset(model, dataset)
        init                                     = get_init(model)
        state_index                              = dict(zip(model._states, range(len(model._states))))
        
        self.tspan       = tspan
        self.t_data      = t_data
        self.y_data      = y_data
        self.s_data      = s_data
        self.exv_names   = exv_names
        self.init        = init
        self.state_index = state_index
        self.model       = model
        
    def get_SSE(self, params_array):
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

                SSE += get_error(ym, yd, sd, idx) if hasattr(ym, '__iter__') else get_error(ym, yd, sd, None)

        return SSE
    
    def __call__(self, params_array):
        return self.get_SSE(params_array)
    
    def __repr__(self):
        return f'{type(self).__name__}<model: {self.model.model_key}>'
    
    def __str__(self):
        return self.__repr__()
        
###############################################################################
#Preprocessing
###############################################################################
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
        if len(tspan_) > 1:
            tspan[scenario] = tspan_  
        else:
            raise ValueError('tspan cannot have only 1 time point.')
        
        for var in t_data[scenario]:
            start, stop = t_data[scenario][var]
            indices_    = indices[start:stop]
            
            #Reassign with the indices
            t_data[scenario][var] = indices_

    return tspan, t_data, y_data, s_data, exv_names

def get_init(model):
    init = {k: v.values for k, v in model.states.T.to_dict('series').items()}
    return init

