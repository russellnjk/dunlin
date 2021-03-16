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
    sim_args   = sim.get_sim_args(model_data)
    return model_data, sim_args

###############################################################################
#Main Algorithm
###############################################################################
def evaluate_objectives(sim_args):
    simulation_results = sim.integrate_models(sim_args)
    obj_results        = parse_simulation_results(simulation_results)
    
    return obj_results

###############################################################################
#Result Parsing
###############################################################################
def parse_simulation_results(simulation_results):
    obj_results = {}
    for model_key, simulation_result in simulation_results.items():
        obj_results[model_key] = parse_simulation_result(simulation_result)
    
    return obj_results
    
def parse_simulation_result(simulation_result):
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
    
    return obj_result

###############################################################################
#Sample Generation
###############################################################################
def sample_dict(base_values, bounds, N=10, analysis_type='sobol', **kwargs):
    #Preprocessing
    problem = {'num_vars' : len(bounds),
               'names'    : list(bounds.keys()),
               'bounds'   : np.array(list(bounds.values())),
               }
    
    #Generate new values
    if analysis_type == 'sobol':          
        new_values   = saltelli.sample(problem, N, **kwargs)
    elif analysis_type == 'fast':   
        new_values   = fast_sampler.sample(problem, N, **kwargs)
    elif analysis_type == 'delta':
        new_values   = latin.sample(problem, N)
    elif analysis_type == 'rbd-fast':
        new_values   = latin.sample(problem, N)
    else:
        raise Exception('Could not find analyzer. analysis_type must be sobol, fast, rbd-fast or delta.')    
    
    #Make DataFrame
    samples = np.zeros((len(new_values), len(base_values)))
    c       = 0
    for i, (key, value) in enumerate(base_values.items()):
        if key in bounds:
            samples[:,i]  = new_values[:,c]
            c            += 1
        else:
            samples[:,i] = value
        
    samples = pd.DataFrame(samples, columns=base_values.keys())  
    return samples

def permute_values(base_values, bounds, N=3, spacing='linear'):
    
    N_ = N if type(N) == dict else {key: N for key in bounds}
    

    samples    = np.zeros( (np.product( tuple(N.values()) ), len(base_values)) )
    
    if spacing == 'linear':
        slices     = [slice(*value, np.complex(0, N[key])) for key, value in bounds.items()]
        new_values = np.mgrid[slices].reshape((len(bounds), -1)).T
    
    elif spacing == 'log':
        slices     = [slice(*np.log(value), np.complex(0, N[key])) for key, value in bounds.items()]
        new_values = np.mgrid[slices].reshape((len(bounds), -1)).T
        new_values = np.exp(new_values)
    
    c = 0
    
    for i, (key, value) in enumerate(base_values.items()):
        if key in bounds:
            samples[:,i]  = new_values[:,c]
            c            += 1
        else:
            samples[:,i] = value
    
    samples = pd.DataFrame(samples, columns=base_values.keys())
    return samples



if __name__ == '__main__':
    base_values = {'a': 1, 'b': 1, 'c': 1}
    bounds      = {'a': np.array([0,   10]),
                   'b': np.array([0, 10])
                   }
    N           = {'a': 3, 'b': 5}
    
    r = permute_values(base_values, bounds, N)
    