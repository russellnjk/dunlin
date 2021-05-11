import numpy   as np
import pandas  as pd
from   pathlib import Path
from SALib.sample import saltelli, fast_sampler, latin

###############################################################################
#Non-Standard Imports
###############################################################################
try:
    import dunlin.model_handler    as mh
    import dunlin.simulation       as sim
    import dunlin._utils_plot.axes as uax
except Exception as e:
    if Path.cwd() == Path(__file__).parent:
        import model_handler    as mh
        import simulation       as sim
        import _utils_plot.axes as uax
    else:
        raise e

###############################################################################
#Globals
###############################################################################
figure        = uax.figure
colors        = uax.colors
palette_types = uax.palette_types
fs            = uax.fs
make_AX       = uax.make_AX
wrap_axfig    = uax.wrap_axfig
scilimit      = uax.scilimit
truncate_time = uax.truncate_time
save_figs     = uax.save_figs

###############################################################################
#High-Level Functions
###############################################################################

###############################################################################
#.ini Parsing
###############################################################################
def read_ini(filename):
    '''
    Reads a .ini file and extracts model data relevant to simulation. Returns a 
    dictionary of keyword arguments that can be passed into downstream functions.

    Parameters
    ----------
    filename : str or Path-like
        The file to be read.

    Returns
    -------
    model_data : dict
        The data from the file read using model_handler.read_ini.
    sim_args : dict
        A dict in the form: 
        {<model_key>: {'model'     : <Model>, 
                       'exvs': <exvs>}
        }
        where <exvs> is a dict of <exv_name>: <exv_function> 
        pairs. 
    
    Notes
    -----
    Passes the model_data variable to get_sim_args so as to obtain sim_args
    
    See Also
    --------
    get_sim_args
    '''
    model_data = mh.read_ini(filename)
    sim_args   = get_sim_args(model_data)
    return model_data, sim_args

def get_sim_args(model_data):
    sim_args = {}
    for key, value in model_data.items():
        sim_args[key] = {'model'      : value['model']
                         }
        
        if 'combinations' in value:
            combinations = value['combinations']
            model        = value['model']
            spacing      = value.get('combination_spacing', 'linear')
            permute_params(model, combinations, spacing)
            
        if 'args' in value:
            sim_args['args'] = value['args']
    
    return sim_args

###############################################################################
#Main Algorithm
###############################################################################
def evaluate_exvs(sim_args, collect_index, goals=None):
    simulation_results = sim.integrate_models(sim_args)
    exv_results        = collect_exv_results(simulation_results, collect_index, goals)
    
    return simulation_results, exv_results

###############################################################################
#Result Parsing
###############################################################################
def collect_exv_results(simulation_results, collect_index=None, goals=None):
    '''
    Iterates through the simulation results and collects the exvs in collect_index 
    into a DataFrame. The DataFrames for the models are organized into a dictionary. 
    If the goals argument is used, the optimized result(s) is 
    returned in the second argument.

    Parameters
    ----------
    simulation_results : dict
        A simulation result data structure as given by dunlin.simulation.
    collect_index : dict of list or dict of tuple, optional
        A dictionary where the keys are the model names and the values are lists 
        of exvs to collect which return a single value. If None, it is assumed 
        that all exvs are to be collected. The default is None.
    goals : {'max', 'min', callable, dict of callable}, optional
        If the goals argument is set to 'max', the second return value is a dict 
        where the keys are the collected exvs and values are the indices of the 
        row(s) with the largest value. The opposite occurs if 'min' is used. Users 
        can customize the behaviour by passing a function. Each column of the 
        DataFrame will be passed into the function. If exv needs to be treated 
        differently, a dictionary can be used where the keys are exvs and the 
        values are functions. If None is used, the second return value is None. 
        The default is None.
        
    Returns
    -------
    exv_results : dict
        A dictionary where the keys are the model names and the values are tuples 
        where the first element is a DataFrame and the second is the optimal 
        result(s).
    '''
    exv_results = {}
    for model_key, simulation_result in simulation_results.items():
        if type(goals) == str or callable(goals):
            goals_ = goals
        elif type(goals) == dict:
            goals_ = goals.get(model_key, None) 
        elif goals is None:
            goals_ = None    
        else:
            raise ValueError('Invalid value for goals: {}'.format(goals))
        
        collect_index_         = collect_index[model_key] if type(collect_index) == dict else collect_index
        exv_results[model_key] = collect_exv_result(simulation_result, collect_index_, goals_)
    
    return exv_results
    
def collect_exv_result(simulation_result, collect_index=(), goals=None):
    '''
    Iterates through the simulation result and collects the exvs in collect_index 
    into a DataFrame. If the goals argument is used, the optimal result(s) is 
    returned in the second argument.

    Parameters
    ----------
    simulation_result : tuple
        A simulation result data structure as given by dunlin.simulation.
    collect_index : list or tuple, optional
        A list of exvs to collect which return a single value. If None or empty 
        it is assumed that all exvs are to be collected.The default is ().
    goals : {'max', 'min', callable, dict of callable}, optional
        If the goals argument is set to 'max', the second return value is a dict 
        where the keys are the collected exvs and values are the indices of the 
        row(s) with the largest value. The opposite occurs if 'min' is used. Users 
        can customize the behaviour by passing a function. Each column of the 
        DataFrame will be passed into the function. If exv needs to be treated 
        differently, a dictionary can be used where the keys are exvs and the 
        values are functions. If None is used, the second return value is None. 
        The default is None.

    Returns
    -------
    exv_result : pandas.DataFrame
        A table of exvs where the first level of the index is the scenario and 
        the second is the estimate.
    best : dict
        If the goals argument is set to 'max', the second return value is a dict 
        where the keys are the collected exvs and values are the indices of the 
        row(s) with the largest value. The opposite occurs if 'min' is used. Users 
        can customize the behaviour by passing a function. Each column of the 
        DataFrame will be passed into the function. If exv needs to be treated 
        differently, a dictionary can be used where the keys are exvs and the 
        values are functions. If None is used, the second return value is None. 
    '''
    exv_result        = {}
    
    for scenario, scenario_result in simulation_result.items():
        for estimate, estimate_result in scenario_result.items():
            table, exvs = estimate_result
            for exv_key, exv_val in exvs.items():
                
                if collect_index:
                    if type(collect_index) == str and exv_key != collect_index:
                        continue
                    elif exv_key not in collect_index:
                        continue
                elif not collect_index:
                    #Check value
                    try:
                        float(exv_val)
                    except:
                        msg = 'The exv value indexed under {} did not return a numerical value. Check the return type of the exv function.'
                        raise TypeError(msg.format(exv_key))
                
                #Make key
                new_key = (scenario, estimate)
                
                #Store result
                exv_result.setdefault(exv_key, {})[new_key] = exv_val
    
    exv_result             = pd.DataFrame.from_dict(exv_result)
    exv_result.index.names = ['scenario', 'estimate']
    
    if goals:
        if type(goals) == dict:
            best = {}
            for key, value in goals:
                if value == 'max':
                    best[key] = exv_result[key].idxmax()
                elif value == 'min':
                    best[key] = exv_result[key].idxmin()
                elif callable(value):
                    best[key] = value(exv_result[key])
                else:
                    raise ValueError('Values in goals must be "max", "min" or a callable.')
        elif goals == 'max':
            best = exv_result.idxmax().to_dict()
        elif goals == 'min':
            best = exv_result.idxmin().to_dict()
        elif callable(goals):
            best = {col: goals(exv_result[col]) for col in exv_result.columns}
        else:
            raise ValueError('goal argument must be "max", "min" or a dict.')
    else:
        best = None
        
    return exv_result, best


###############################################################################
#Sample Generation
###############################################################################
def permute_params(model, combinations, spacing='linear'):
    param_vals = model.param_vals
    if len(param_vals) != 1:
        raise ValueError('Model {} can only have one row in its param_vals.'.format(model.name))
    param_vals_    = dict(zip(param_vals.columns, param_vals.values[0]))
    new_param_vals = permute_values(param_vals_, combinations, spacing)
    
    model.param_vals = new_param_vals
    return new_param_vals

def make_slice(lb, ub, n=3):
    '''
    :meta private:
    '''
    return slice(lb, ub, np.complex(0, n))
    
def permute_values(base_values, combinations, spacing='linear'):
    '''
    :meta private:
    '''
    #Preprocessing
    rows    = np.product([value[2] if len(value) == 3 else 3 for value in combinations.values()])
    samples = np.zeros( (int(rows), len(base_values)) )
    
    #Generate samples
    if spacing == 'linear':
        slices     = [make_slice(*value) for key, value in combinations.items()]
        new_values = np.mgrid[slices].reshape((len(combinations), -1)).T
    
    elif spacing == 'log':
        slices     = [make_slice(*np.log10(value[:2]), value[2]) for key, value in combinations.items()]
        new_values = np.mgrid[slices].reshape((len(combinations), -1)).T
        new_values = 10**new_values

    #Assign values
    c = 0
    
    for i, (key, value) in enumerate(base_values.items()):
        if key in combinations:
            samples[:,i]  = new_values[:,c]
            c            += 1
        else:
            samples[:,i] = value
    
    if np.isnan(samples).any():
        raise ValueError('Detected NaN values in samples. Check the values used in the combinations and spacing arguments.')
        
    #Convert to DataFrame
    samples = pd.DataFrame(samples, columns=base_values.keys())
    return samples

if __name__ == '__main__':
    # # #Test permutation
    # #Case 1: Linear
    # base_values  = {'a': 1, 'b': 1, 'c': 1}
    # combinations = {'a': np.array([0, 10, 3]),
    #                 'b': np.array([0, 10, 5])
    #                 }
    
    # r = permute_values(base_values, combinations)
    # assert all(r['a'] == [0]*5 + [5]*5 + [10]*5)
    # assert all(r['b'] == [0, 2.5, 5, 7.5, 10]*3)
    # assert all(r['c'] == 1)
    
    # #Case 2: Logarithmic
    # combinations = {'a': np.array([0.1, 10, 3]),
    #                 'b': np.array([0.1, 10, 3])
    #                 }
    
    # r = permute_values(base_values, combinations, 'log')
    # assert all(np.isclose(r['a'], [0.1]*3 + [1]*3 + [10]*3))
    # assert all(np.isclose(r['b'], [0.1, 1, 10]*3))
    # assert all(r['c'] == 1)
    
    # #Test param reassignment
    # model_data = mh.read_ini('_test/TestCombinatorial_1.ini')
    # sim_args   = get_sim_args(model_data)
    # model_1    = sim_args['model_1']['model']
    # param_vals = model_1.param_vals
    
    # assert len(param_vals)          == 9
    # assert all(param_vals['synx']   == 0.08)
    # assert all(param_vals['syny']   == 0.08)
    # assert not all(param_vals['Jx'] == 5e-4)
    # assert not all(param_vals['Jy'] == 5e-2)
    
    # #Test exv evaluation
    # simulation_results, exv_results = evaluate_exvs(sim_args)
    #
    # exv_model_1 = exv_results['model_1'][0]
    # vals        = exv_model_1.values.flatten()
    # assert len(exv_model_1) == 18
    # assert all(np.isclose(vals[:9], 0, atol=1e-4))
    # assert all(np.isclose(vals[9:], [0.00132, 11.206, -2.369, 0.001, 10.702, -2.25, 0.001, 10.214, -2.13], atol=1e-3))
    
    # #Test reading .ini
    # #Case 1: Without spacing specifications
    # model_data, sim_args = read_ini('_test/TestCombinatorial_1.ini')
    # assert 'combinations' in model_data['model_1']
    
    # model_1    = sim_args['model_1']['model']
    # param_vals = model_1.param_vals
    # assert len(param_vals) == 15
    
    #Case 2: With spacing specifications
    model_data, sim_args = read_ini('_test/TestCombinatorial_2.ini')
    assert 'combination_spacing' in model_data['model_1']
    assert model_data['model_1']['combination_spacing'] == 'log'
    
    model_1    = sim_args['model_1']['model']
    param_vals = model_1.param_vals
    assert any(np.isclose(param_vals['Jx'], 0.000316, atol=1e-6))
    