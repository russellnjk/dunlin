import numpy   as np
import pandas  as pd
from   pathlib import Path
from SALib.sample import saltelli, fast_sampler, latin

###############################################################################
#Non-Standard Imports
###############################################################################
try:
    import dunlin.model_handler          as mh
    import dunlin.optimize               as opt
    import dunlin.wrapSSE                as ws
    import dunlin.simulation             as sim
    import dunlin._utils_plot.utils_plot as utp
except Exception as e:
    if Path.cwd() == Path(__file__).parent:
        import model_handler          as mh
        import optimize               as opt
        import wrapSSE                as ws
        import simulation             as sim
        import _utils_plot.utils_plot as utp
    else:
        raise e

###############################################################################
#Globals
###############################################################################
colors        = utp.colors
palette_types = utp.palette_types
fs            = utp.fs

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
                       'objectives': <objectives>}
        }
        where <objectives> is a dict of <objective_name>: <objective_function> 
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
        sim_args[key] = {'model'      : value['model'],
                         'objectives' : value.get('objectives')
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
def evaluate_objectives(sim_args, goals=None):
    simulation_results = sim.integrate_models(sim_args)
    obj_results        = collect_objective_results(simulation_results, goals)
    
    return simulation_results, obj_results

###############################################################################
#Result Parsing
###############################################################################
def collect_objective_results(simulation_results, goals=None):
    obj_results = {}
    for model_key, simulation_result in simulation_results.items():
        if type(goals) == str or callable(goals):
            goals_ = goals
        elif type(goals) == dict:
            goals_ = goals.get(model_key, None) 
        elif goals is None:
            goals_ = None    
        else:
            raise ValueError('Invalid value for goals: {}'.format(goals))
        
        obj_results[model_key] = collect_objective_result(simulation_result, goals_)
    
    return obj_results
    
def collect_objective_result(simulation_result, goals=None):
    obj_result        = {}
    
    for scenario, scenario_result in simulation_result.items():
        for estimate, estimate_result in scenario_result.items():
            table, objs = estimate_result
            for obj_key, obj_val in objs.items():
                
                #Check value
                try:
                    float(obj_val)
                except:
                    msg = 'The objective value indexed under {} did not return a numerical value. Check the return type of the objective function.'
                    raise TypeError(msg.format(obj_key))
                
                #Make key
                new_key = (scenario, estimate)
                
                #Store result
                obj_result.setdefault(obj_key, {})[new_key] = obj_val
    
    obj_result             = pd.DataFrame.from_dict(obj_result)
    obj_result.index.names = ['scenario', 'estimate']
    
    if goals:
        if type(goals) == dict:
            best = {}
            for key, value in goals:
                if value == 'max':
                    best[key] = obj_result[key].idxmax()
                elif value == 'min':
                    best[key] = obj_result[key].idxmin()
                elif callable(value):
                    best[key] = value(obj_result[key])
                else:
                    raise ValueError('Values in goals must be "max", "min" or a callable.')
        elif goals == 'max':
            best = obj_result.idxmax().to_dict()
        elif goals == 'min':
            best = obj_result.idxmin().to_dict()
        elif callable(goals):
            best = {col: goals(obj_result[col]) for col in obj_result.columns}
        else:
            raise ValueError('goal argument must be "max", "min" or a dict.')
    else:
        best = None
        
    return obj_result, best


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
    return slice(lb, ub, np.complex(0, n))
    
def permute_values(base_values, combinations, spacing='linear'):

    #Preprocessing
    rows    = np.product([value[2] if len(value) == 3 else 3 for value in combinations.values()])
    samples = np.zeros( (int(rows), len(base_values)) )
    
    #Generate samples
    if spacing == 'linear':
        slices     = [make_slice(*value) for key, value in combinations.items()]
        new_values = np.mgrid[slices].reshape((len(combinations), -1)).T
    
    elif spacing == 'log':
        slices     = [make_slice(*np.log(value[:2]), value[2]) for key, value in combinations.items()]
        new_values = np.mgrid[slices].reshape((len(combinations), -1)).T
        new_values = np.exp(new_values)
    
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
    
    # #Test objective evaluation
    # simulation_results, obj_results = evaluate_objectives(sim_args)
    #
    # obj_model_1 = obj_results['model_1'][0]
    # vals        = obj_model_1.values.flatten()
    # assert len(obj_model_1) == 18
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
    