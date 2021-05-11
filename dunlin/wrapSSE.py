import numpy   as     np
from   numba   import jit
from   pathlib import Path

###############################################################################
#Non-Standard Imports
###############################################################################
try:
    import dunlin._utils_model.integration as itg
    import dunlin.model_handler            as mh
    import dunlin.simulation               as sim
except Exception as e:
    if Path.cwd() == Path(__file__).parent:
        import _utils_model.integration as itg
        import model_handler            as mh
        import simulation               as sim
    else:
        raise e

###############################################################################
#High-level Preprocessing
###############################################################################
def preprocess_SSE(models, exp_data):
    '''
    Accepts models and experimental data and returns an index of the parameter 
    names and a function which accepts an array of parameter values and returns 
    the SSE associated with the models in the models argument.

    Parameters
    ----------
    models : 'dict of Models'
        DESCRIPTION.
    exp_data : dict
        A dict in the form {<model_key>: dataset} where each dataset is in turn 
        a dict in the form {(<state>, <scenario>, <'Time' or 'Data'>): <values>} 
        where state is a state in the corresponding model. Every data measurement 
        must have a corresponing time measurement. For example, ('x', 0, 'Data') 
        must be accompanied by ('x', 0, 'Time').

    Returns
    -------
    param_names : dict
        An index of the parameter names for the model. Required for downstream 
        processing.
    get_SSE : function
        A function that accepts a numpy.ndarray of parameters and returns the 
        SSE for all the models. The sign of the SSE is inverted due to requirements 
        of the Metropolitan - Hastings algorithm.
    '''
    #Check models
    if type(models) in [list, tuple]:
        models_ = {model.name: model for model in models}
    elif type(models) == dict:
        models_ = models
    else:
        msg = 'models argument should be a dict of models indexed by their names. Received {} instead.'
        raise TypeError(msg.format(type(models)))
    
    model_types = [type(model) for model in models_.values()]
    if not all( [model_type == mh.Model for model_type in model_types] ):
        msg = 'All models must be an instance of the Model class. Received {} instead.'
        raise TypeError(msg.format(model_types))
    
    param_names, param_index = get_param_index(models_)
    
    sse_calcs = {key: wrap_get_SSE_dataset(model, exp_data[key]) for key, model in models_.items()}
    
    get_SSE = wrap_get_SSE(param_index, sse_calcs)
    
    return param_names, get_SSE

###############################################################################
#SSE Calculation
###############################################################################
def wrap_get_SSE(param_index, sse_calcs):
    '''
    Returns a function that accepts an array of parameters and returns the SSE 
    summed across all the models associated with sse_calcs.

    Parameters
    ----------
    param_index : dict
        An index of the parameter names for the model. Required for downstream 
        processing.
    sse_calcs : dict
        A dict in the form {<model_key>: <get_SSE>}.

    Returns
    -------
    get_SSE : function
        A function that accepts a numpy.ndarray of parameters and returns the 
        SSE for all the models. The sign of the SSE is inverted due to requirements 
        of the Metropolitan - Hastings algorithm.
    '''
    def get_SSE(params_array: np.ndarray):
        SSE = 0
        
        for model_key, indices in param_index.items():
            params_array_  = params_array[indices]
            SSE           += sse_calcs[model_key](params_array_)
            
        return SSE
    return get_SSE

def wrap_get_SSE_dataset(model, dataset, custom_func=None):
    '''
    Returns a function that accepts an array of parameters and returns the SSE 
    associated with the model provided in the model argument. 

    Parameters
    ----------
    model : Model
        A Model object for numerical integration.
    dataset : dict
        A dict in the form {(<state>, <scenario>, <'Time' or 'Data'>): <values>} 
        where state is a state in the corresponding model. Every data measurement 
        must have a corresponing time measurement. For example, ('x', 0, 'Data') 
        must be accompanied by ('x', 0, 'Time').

    Returns
    -------
    get_SSE_dataset : function
        A function that accepts a numpy.ndarray of parameters and returns the 
        SSE for the model. The sign of the SSE is inverted due to requirements 
        of the Metropolitan - Hastings algorithm.
    
    Notes
    -----
    Makes an internal call to make_fixed_args. Sign is inverted due to requirements 
    of MCMC.
    '''
    y_data, s_data, t_indices, int_args, state_index, init_dict, inputs_dict = make_fixed_args(model, dataset)
    
    def get_SSE_dataset(params):
        SSE = 0
        
        for scenario, init in init_dict.items():
            
            inputs           = inputs_dict[scenario]
            y_model, t_model = itg.piecewise_integrate(init     = init, 
                                                       params   = params, 
                                                       inputs   = inputs, 
                                                       scenario = scenario, 
                                                       **int_args
                                                       )
            
            if custom_func:
                SSE_ = custom_func(y_data, y_model, t_model, scenario)
                SSE += SSE_
                continue
            
            for state, y_d in y_data[scenario].items():    
                row = state_index[state]
                
                if callable(row):
                    t_key = (state, scenario, 'Time')
                    
                    if t_key in t_indices:
                        cols     = t_indices[(state, scenario, 'Time')]
                        p, u     = sim.get_matrices(int_args['function'], int_args['tspan'], y_model, params, inputs, scenario, modify=int_args.get('modify'), args=int_args.get('args', ()))
                        y_m      = row(t_model[cols], y_model[:,cols], p[:,cols], u[:,cols])
                    else:
                        p, u = sim.get_matrices(int_args['function'], int_args['tspan'], y_model, params, inputs, scenario, modify=int_args.get('modify'), args=int_args.get('args', ()))
                        y_m = row(t_model, y_model, p, u)
                    
                    if type(y_m) in [tuple, list]:
                        y_m = y_m[1]
                    elif type(y_m) == dict:
                        y_m = y_m['y']
                    
                else:
                    cols = t_indices[(state, scenario, 'Time')]
                    y_m  = y_model[row, cols]
                    
                SSE = get_SSE_vector(y_d, y_m, s_data.get(state, 1), SSE)
        return -SSE
    
    return get_SSE_dataset

@jit(nopython=True)
def get_SSE_vector(data_vector, model_vector, normalization=1, curr_val=0):
    '''
    :meta private:
    '''
    return np.sum((data_vector - model_vector)**2)/normalization + curr_val

###############################################################################
#Fixed Argument Pre-Processing
###############################################################################
def make_fixed_args(model, dataset):
    '''
    Extracts arguments that remain constant during the curve-fitting process. 

    Parameters
    ----------
    model : Model
        A Model object for numerical integration.
    dataset : dict
        A dict in the form {(<state>, <scenario>, <'Time' or 'Data'>): <values>} 
        where state is a state in the corresponding model. Every data measurement 
        must have a corresponing time measurement. For example, ('x', 0, 'Data') 
        must be accompanied by ('x', 0, 'Time').

    Returns
    -------
    y_data : dict
        A dict of arrays for SSE calculation.
    t_indices : dict
        Indices for extracting the state values at the correct timepoints.
    int_args : dict
        Miscellaneous arguments for integration such as solver settings etc..
    state_index : dict
        An index of the model's states.
    init_dict : dict
        Initial values.
    inputs_dict : dict
        Input values.
    
    :meta private:
    '''
    #State and input pre-processing
    state_index = index_list(model.states) 
    inputs_dict = {i: df.values for i, df in model.input_vals.groupby(level=0)}
    init_dict   = dict(zip(model.init_vals.index, model.init_vals.values))
    
    if model.exvs:
        state_index = {**state_index, **model.exvs}
        
    y_data = {}
    t_data = {}
    s_data = {}
    n_data = {}
    for key, value in dataset.items():
        if key[1] == 'Variance':
            s_data[key[0]] = 2*value
        elif key[2] == 'Data':
            y_data.setdefault(key[1], {})[key[0]] = value
            
            n_data.setdefault(key[0], 0) 
            n_data[key[0]] += len(value)
        elif key[2] == 'Time':
            y_data.setdefault(key[1], {}).setdefault(key[0], 0)
            t_data[key] = value
        else:
            raise ValueError(f'Key not recognized {key}')
    
    for state, npoints in n_data.items():
        if state in s_data:
            s_data[state] /= npoints
        else:
            s_data[state] = npoints

    #Time point processing
    t_indices, sse_tspan = get_t_indices_and_tspan(model.tspan, t_data)
    
    int_args = {'function' : model.func,
                'tspan'    : sse_tspan,
                'modify'   : model.modify,
                }
    int_args.update(model.solver_args)
    
    return y_data, s_data, t_indices, int_args, state_index, init_dict, inputs_dict

###############################################################################
#Time Point Pre-Processing
###############################################################################
def get_t_indices_and_tspan(user_tspan, t_data):
    '''
    For pre-processing the fixed arguments for curve-fitting.
    
    :meta private:
    '''
    #Get time points in dataset and time points at the ends of each segment
    tp_data = list(t_data.values())
    tp_seg  = [user_tspan[0][0]] + [segment[-1] for segment in user_tspan]

    #Split time points according to segments
    sse_tspan = [[] for i in range(len(user_tspan))]

    tpoints   = np.unique(np.concatenate(tp_data + [tp_seg]) )
    bins      = np.digitize(tpoints, tp_seg, right=True) 
    bins[0]   = 1
    curr      = 1
    t2index   = {}

    #Check len of tspan
    if tp_seg[-1] < tpoints[-1]:
        raise Exception('The last timepoint in the data exceeds the time span of the model. Check the tspan you are using in your model.')
    
    for i, tp in enumerate(tpoints):
        
        b = bins[i]
        
        t2index.setdefault(tp, i+b-1)
        if b > curr:
            sse_tspan[b-1].append(tpoints[i-1])
            curr += 1

        sse_tspan[b-1].append(tp)
        
    sse_tspan = [np.array(x) for x in sse_tspan]
    
    #Get indices for each time point
    t_indices = {}
    for key, value in t_data.items():
        t_indices[key] = np.array([t2index[tpoint] for tpoint in value])

    return t_indices, sse_tspan
            
def find_indices(tpoints, t_array):
    '''
    :meta private:
    '''
    indices = []
    print(tpoints)
    print(t_array)
    assert False
    for t in t_array:
        index = np.where(t == tpoints)
        if len(index) == 0:
            raise Exception(str(t)+' not in t_model.')
        else:
            indices.append(index[0][0])
    return np.array(indices)

###############################################################################
#Other Pre-Processing
###############################################################################
def index_list(lst):
    '''
    :meta private:
    '''
    return {x: i for i,x in enumerate(lst)}

def get_param_index(models):
    '''
    :meta private:
    '''
    param_names = {}
    i           = 0
    param_index = {}  
    
    for key, model in models.items():
        param_index[key] = []
        for param in model.params:
            
            if param not in param_names:
                param_names[param] = i
                i+= 1
            
            param_index[key].append(param_names[param])
    
    param_index = {key: np.array(value) for key, value in param_index.items()}
    
    return param_names, param_index

if __name__ == '__main__':
    #Data set level tests
    #Make data
    time1   = np.linspace(0,  1, 21)
    time2   = np.linspace(0,  2, 21)
    y_data1 = np.e**(-np.linspace(0, 1, 21))
    y_data2 = 2 -2*np.e**(-np.linspace(0, 2, 21))
    dataset = {('x', 0, 'Data') : y_data1,
               ('x', 0, 'Time') : time1,
               ('x', 1, 'Data') : y_data2,
               ('x', 1, 'Time') : time2,
               ('w', 0, 'Data') : y_data1,
               ('w', 0, 'Time') : time1,
               ('w', 1, 'Data') : y_data2,
               ('w', 1, 'Time') : time2,               
               }
    
    #Read model
    model_data = mh.read_ini('_test/TestCurveFit_1.ini')
    model      = model_data['model_1']['model']

    #Preprocessing
    state_index = index_list(model.states)
    param_index = index_list(model.params)
    inputs_dict = {i: df.values for i, df in model.input_vals.groupby(level=0)}
    init_dict   = dict(zip(model.init_vals.index, model.init_vals.values))
    tspan       = model.tspan
    func        = model.func
    
    #Split Data and Time
    y_data = {key: value for key, value in dataset.items() if key[2]=='Data'}
    t_data = {key: value for key, value in dataset.items() if key[2]=='Time'}
    
    #Test time point processing
    t_indices, sse_tspan = get_t_indices_and_tspan(model.tspan, t_data)
    
    assert len(sse_tspan)    == 2
    assert len(sse_tspan[0]) == 21
    assert len(sse_tspan[1]) == 11
    assert sse_tspan[0][-1]  == sse_tspan[1][0]
    assert all( [len(value.shape) == 1 for value in t_indices.values()] )
    assert all( [type(x) == np.ndarray for x in sse_tspan] )
    
    all_t   = np.concatenate(sse_tspan)
    for key, t_data_ in t_data.items():
        indices = t_indices[key]
        assert all(t_data_ == all_t[indices])
        
    #Test integration
    params_array     = np.array([1, 1, 2])
    
    # Single scenario integration
    scenario         = 1
    inputs           = inputs_dict[scenario]
    init             = init_dict[scenario]
    y_model, t_model = itg.piecewise_integrate(func, sse_tspan, init, params_array, inputs, scenario)
    assert all(t_model == all_t)
    
    #Test SSE Calculation
    state    = 'x'
    y_d      = y_data[(state, scenario, 'Data')]
    cols     = t_indices[(state, scenario, 'Time')]
    row      = state_index['x']
    y_m      = y_model[row, cols]
    assert y_m.shape == y_d.shape
    
    SSE0     = get_SSE_vector(y_d, y_m)
    assert np.isclose(SSE0, 0, atol=1e-5)
    
    #Test dataset level SSE calculation
    get_SSE_dataset = wrap_get_SSE_dataset(model, dataset)
    SSE1            = get_SSE_dataset(params_array)
    assert np.isclose(SSE1, 0, atol=1e-5)    
    
    #Multi-model tests
    #Make data
    time1   = np.linspace(0,  1, 21)
    time2   = np.linspace(0,  2, 21)
    y_data1 = np.e**(-np.linspace(0, 1, 21))
    y_data2 = 2 -2*np.e**(-np.linspace(0, 2, 21))
    dataset = {('x', 0, 'Data') : y_data1,
                ('x', 0, 'Time') : time1,
                ('x', 1, 'Data') : y_data2,
                ('x', 1, 'Time') : time2,
                ('w', 0, 'Data') : y_data1,
                ('w', 0, 'Time') : time1,
                ('w', 1, 'Data') : y_data2,
                ('w', 1, 'Time') : time2,               
                }
    exp_data = {'model_1': dataset,
                'model_2': dataset
                }
    
    #Read model
    model_data = mh.read_ini('_test/TestCurveFit_2.ini')
    models     = {key: value['model'] for key, value in model_data.items()}
    
    #Make guess
    params_array = np.array([1, 1, 2, 1, 0])
    
    #Test param indexing
    param_names, param_index = get_param_index(models)
    assert all(param_index['model_1'] == [0, 1, 2])
    assert all(param_index['model_2'] == [0, 3, 4])
    
    params_array2 = params_array[param_index['model_2']]
    assert all(params_array2 == [1, 1, 0])  
    
    #Test get_SSE
    params_array = np.array([1, 1, 2, 1, 0])
    sse_calcs = {key: wrap_get_SSE_dataset(model, exp_data[key]) for key, model in models.items()}
    get_SSE   = wrap_get_SSE(param_index, sse_calcs)
    SSE0      = get_SSE(params_array)
    assert np.isclose(SSE0, -44.49428, atol=5e-2)
    
    params_array                       = np.array([1, 1, 2, 1, 2])
    models['model_2'].init_vals.loc[1] = 0
    
    sse_calcs = {key: wrap_get_SSE_dataset(model, exp_data[key]) for key, model in models.items()}
    get_SSE   = wrap_get_SSE(param_index, sse_calcs)
    SSE1      = get_SSE(params_array)
    assert np.isclose(SSE1, 0, atol=5e-2)
    
    #Test single-step preprocessing
    param_names, get_SSE = preprocess_SSE(models, exp_data)
    SSE2                 = get_SSE(params_array)
    assert np.isclose(SSE2, 0, atol=5e-2)
    
    #Read model
    model_data2 = mh.read_ini('_test/TestCurveFit_3.ini')
    models2     = {key: value['model'] for key, value in model_data2.items()}
    
    param_names, get_SSE = preprocess_SSE(models2, exp_data)
    SSE3                 = get_SSE(params_array)
    assert np.isclose(SSE3, 0, atol=5e-2)
    
   