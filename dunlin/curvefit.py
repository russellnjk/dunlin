import numpy   as np
import pandas  as pd
from   pathlib import Path

###############################################################################
#Non-Standard Imports
###############################################################################
try:
    import dunlin.model_handler                as mh
    import dunlin._utils_optimization.optimize as opt
    import dunlin._utils_optimization.wrapSSE  as ws
    import dunlin.simulation                   as sim
    import dunlin._utils_plot.utils_plot       as utp
    from   dunlin._utils_optimization.dataparser import read_csv
except Exception as e:
    if Path.cwd() == Path(__file__).parent:
        import model_handler                as mh
        import _utils_optimization.optimize as opt
        import _utils_optimization.wrapSSE  as ws
        import simulation                   as sim
        import _utils_plot.utils_plot       as utp
        from   _utils_optimization.dataparser import read_csv
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
def integrate_and_plot(plot_index, sim_args, posterior=None, guesses=None, exp_data=None, AX=None, **line_args):

    figs, AX1                    = (None, AX) if AX else utp.make_AX(plot_index)
    posterior_simulation_results = None
    guess_simulation_results     = None
    
    if posterior is not None:
        _, AX1, posterior_simulation_results = integrate_and_plot_posterior(plot_index, sim_args, posterior, AX1, **line_args)
    
    if guesses is not None:
        if type(guesses) == dict:
            guesses_ = pd.DataFrame.from_dict(guesses, orient='index')
        else:
            guesses_ = guesses
        
        guess_line_args = {**line_args, **{'linestyle': ':', 'label': '_nolabel'}}
        
        _, AX1, guess_simulation_results = integrate_and_plot_posterior(plot_index, sim_args, guesses_, AX1, **guess_line_args)
    
    if exp_data is not None:
        exp_line_args = {**line_args, **{'linestyle': '', 'marker': '+', 'label': '_nolabel'}}
        
        _, AX1 = plot_exp_data(plot_index, exp_data, AX1, **exp_line_args)
    
    return figs, AX1, posterior_simulation_results, guess_simulation_results
        
def integrate_and_plot_posterior(plot_index, sim_args, posterior, AX=None, **line_args):
    
    simulation_results = integrate_posteriors(sim_args, posterior)
    figs, AX1          = plot_posterior(plot_index, simulation_results, AX, **line_args)
    
    return figs, AX1, simulation_results 

###############################################################################
#.ini Parsing
###############################################################################
def read_ini(filename, sampler='sa'):
    '''
    Reads a .ini file and extracts the arguments required for running curve-fitting.

    Parameters
    ----------
    filename : str or Path-like
        Name of file to open.
    sampler : {'sa'}
        The type of sampler to be used. The default is 'sa'.

    Returns
    -------
    model_data : dict
        The data from the file read using model_handler.read_ini.
    cf_args: dict
        A dictionary of keyword arguments.
    guesses: dict 
        Initial guesses for parameter estimation.     
    '''
    model_data = mh.read_ini(filename)
    
    if sampler == 'sa':
        return (model_data,) + get_sa_args(model_data)
    else:
        raise NotImplementedError('Not implemented yet!')

def get_sa_args(model_data, exp_data=None):   
    '''
    Extracts the arguments required for running simulated annealing.

    Parameters
    ----------
    model_data : dict
        The data from the file read using model_handler.read_ini.

    Returns
    -------
    guesses : dict
        A dictionary of initial guesses.
    cf_args : dict
        A dictionary of keyword arguments.
    '''
    models     = {}
    priors     = {}
    bounds     = {}
    guesses    = {}
    step_size  = {}
    iterations = []

    for key, value in model_data.items():
        model          = value['model']
        models[key]    = model
        param_vals     = model.param_vals.to_dict('index')
           
        for k, v in param_vals.items():
            guesses.setdefault(k, v).update(v)
            
        priors.update( value.get('priors', {}) )
        bounds.update( value.get('param_bounds', {}) )
        
        if 'cf_iterations' in value:
            iterations.append(value['cf_iterations'])
        
        if 'step_size' in value:
            step_size.update(value['step_size'])
        
    iterations = max(iterations) if iterations else 10000
    
    cf_args = {'models'     : models,
               'priors'     : priors,
               'bounds'     : bounds,
               'iterations' : iterations,
               'step_size'  : step_size,
               'exp_data'   : exp_data
               }
    
    return guesses, cf_args

###############################################################################
#Simulated Annealing
###############################################################################
def simulated_annealing(models, exp_data, guess, step_size, priors=None, bounds=None, iterations=10000, SA=True, callback=None):
    '''
    Performs simulated annealing to find the posterior distribution of the parameters.    

    Parameters
    ----------
    models : dict of Model
        A dict of Model objects.
    exp_data : dict
        A dict in the form {<model_key>: dataset} where each dataset is in turn 
        a dict in the form {(<state>, <scenario>, <'Time' or 'Data'>): <values>} 
        where state is a state in the corresponding model. Every data measurement 
        must have a corresponing time measurement. For example, ('x', 0, 'Data') 
        must be accompanied by ('x', 0, 'Time').
    guess : dict or pd.Series
        Initial guess of parameters.
    step_size : dict
        A dict of step sizes to take for each parameter. A parameters will not be 
        sampled if its step size is 0 or not included in this argument.
    priors : dict, optional
        A dict in the form {<parameter_name>: [<mean>, <st_dev>]}. Represents 
        Gaussian priors. The default is None. 
    bounds : dict, optional
        A dict in the form {<parameter_name>: [<lower>, <upper>]}. The default 
        is None.
    iterations : int, optional
        The number of iterations to take. The default is 10000.
    SA : bool, optional
        If True, the temperature parameter of the acceptance criterion changes 
        with each step. If False, the temperature remains constant and the algorithm
        becomes a typial Gibbs sampler. The default is True.
    callback : function or tuple of functions, optional
        A function(s) to be called using func(log_posterior, log_prior, log_likelihood). 
        Keywords are not to be used. The default is None.

    Returns
    -------
    opt_result : dict
        Contains a DataFrame of accepted parameters indexed under 'accepted'. Also 
        contains other important variables that were generated internally.
    '''
    opt_args, others = preprocess_simulated_annealing(models, exp_data, guess, step_size, priors, bounds, iterations, SA, callback, absolute_step_size=False)
    
    opt_result             = opt.simulated_annealing(**opt_args)
    columns                = list(others['param_index'].keys())
    opt_result['accepted'] = pd.DataFrame(opt_result['accepted'], columns=columns)
    
    opt_result = {**opt_result, **others}
    return opt_result 

def preprocess_simulated_annealing(models, exp_data, guess, step_size, priors=None, bounds=None, iterations=10000, SA=True, callback=None, absolute_step_size=False):
    '''
    :meta private:
    '''
    #Convert guess
    msg = 'guess must be a dict, pandas.Series or 1-row pandas.DataFrame'
    if type(guess) == dict:
        guess_ = guess
    elif type(guess) == pd.Series:
        guess_ = guess.to_dict()
    elif type(guess) == pd.DataFrame:
        if guess.shape[0] == 1:
            guess_ = guess.to_dict('index')
            guess_ = guess_[next(iter(guess_))]
        else:
            raise Exception(msg)
    else:
        raise Exception(msg)
        
    #Get param_names, get_SSE, get_log_prior, get_log_posterior, check_bounds
    exp_data_            = format_exp_data(exp_data)
    param_names, get_SSE = ws.preprocess_SSE(models, exp_data_)
    
    model_params  = set(param_names.keys())
    guess_params  = set(guess.keys())
    prior_params  = set(priors.keys()) if priors else set()
    bounds_params = set(bounds.keys()) if bounds else set()
    
    check_missing_params(   model_params, guess_params,  name='guess')
    check_unexpected_params(model_params, guess_params,  name='guess')
    check_unexpected_params(model_params, prior_params,  name='priors')
    check_unexpected_params(model_params, bounds_params, name='bounds')
    
    get_log_prior        = opt.wrap_get_log_prior(priors, param_names)
    get_log_posterior    = opt.wrap_get_log_posterior(get_SSE, get_log_prior, callback)
    check_bounds         = opt.wrap_check_bounds(bounds, param_names)
    
    #Convert step_size to transition
    if type(step_size) == dict:
        if len(step_size) == 0:
            msg = 'step_size must have at least one parameter. Received: {}'
            raise ValueError(msg.format(step_size))
            
        step_params  = set(step_size.keys()) 
        check_unexpected_params(model_params, step_params, name='step_size')
        
        transition = opt.wrap_transition(step_size, param_names)
        
    elif callable(step_size):
        transition = step_size
    else:
        msg = 'step_size argument must be a dict or a function. Received {} instead.'
        raise TypeError(msg.format(type(step_size)))
    
    #Check guess falls within bounds
    check_guess_in_bounds(guess, check_bounds, bounds)
        
    #Compile into dict    
    opt_args = {'func'         : get_log_posterior,
                'guess'        : guess,
                'check_bounds' : check_bounds,
                'iterations'   : iterations,
                'transition'   : transition,
                'SA'           : SA
                }
    others   = {'param_index'  : param_names,
                'posterior'    : get_log_posterior, 
                'SSE'          : get_SSE,
                'prior'        : get_log_prior,
                'bounds'       : check_bounds,
                'exp_data'     : exp_data_
                }
    return opt_args, others

###############################################################################
#Parameter Checks
###############################################################################
def check_missing_params(model_params, other_params, name='guess'):
    '''
    :meta private:
    '''
    missing      = [param for param in model_params if param not in other_params]
    if missing:
        raise ParameterError('Missing parameters in {}: {}.'.format(name, missing))
    return

def check_unexpected_params(model_params, other_params, name='guess'):
    '''
    :meta private:
    '''
    unexpected   = [param for param in other_params if param not in model_params]
    if unexpected:
        raise ParameterError('Unexpected parameters in {}: {}.'.format(name, unexpected))

def check_guess_in_bounds(guess, check_bounds, bounds):
    '''
    :meta private:
    '''
    if check_bounds:
        guess_array = np.array(list(guess.values()))
        if not check_bounds(guess_array):
            msg = 'Initial guess of parameter values not within bounds. guess:{}\nbounds:{}'
            raise ParameterError(msg.format(guess, bounds))
        
class ParameterError(Exception):
    '''
    :meta private:
    '''
    pass

###############################################################################
#Data Check
###############################################################################
def format_exp_data(exp_data):
    '''
    :meta private:
    '''
    if type(exp_data) == dict:
        exp_data_ = exp_data
    elif type(exp_data) == pd.DataFrame:
        exp_data_ = exp_data.to_dict()
    else:
        raise TypeError('exp_data must be a dict or pandas.DataFrame.')
      
    formatted = {key: format_dataset(dataset, key) for key, dataset in exp_data_.items()}
    return formatted

def format_dataset(dataset, key):
    '''
    :meta private:
    '''
    msg0 = 'Error in parsing data for {}.\n'.format(key) 
    msg1 = 'keys must be a tuple formatted as (<state>, <scenario>, <"Data" or "Time">)'
    msg2 = 'Missing data for ({}, {}, {})'
    msg3 = 'Mismatched Time/Data lengths for {}, {}'
    msg4 = 'Array for {} must be 1-D but has shape {}'
    
    def check_shape(y_data, time):
        if len(y_data.shape) != 1:
            raise ExperimentalDataError(msg4.format(key, y_data.shape))
        elif len(time.shape) != 1:
            raise ExperimentalDataError(msg4.format(key, time.shape))
    
    def check_key(key, key_):
        if key_ not in dataset:
            raise ExperimentalDataError(msg0 + msg2.format(*key_))
        elif len(dataset[key]) != len(dataset[key_]):
            raise ExperimentalDataError(msg0 + msg3.format(key[0], key[1]))
            
    formatted = {}
    
    for key, value in dataset.items():
        if key in formatted:
            continue
        
        if type(key) != tuple:
            raise ExperimentalDataError(msg0 + msg1)
        elif len(key) != 3:
            raise ExperimentalDataError(msg0 + msg1)
        elif key[2] != 'Data' and key[2] != 'Time':
            raise ExperimentalDataError(msg0 + msg1)

        elif key[2] == 'Data':
            key_ = (key[0], key[1], 'Time')
            
            check_key(key, key_)
            
            y_data = np.array(value)
            time   = np.array(dataset[key_])
            
            check_shape(y_data, time)
            
            formatted[key]  = y_data
            formatted[key_] = time
                
        elif key[2] == 'Time':
            key_ = (key[0], key[1], 'Data')
            
            check_key(key, key_)
            
            time   = np.array(value)
            y_data = np.array(dataset[key_])
            
            check_shape(y_data, time)
            
            formatted[key]  = time
            formatted[key_] = y_data
        
    return formatted
            
class ExperimentalDataError(Exception):
    '''
    :meta private:
    '''
    pass

###############################################################################
#Plotting
###############################################################################
def plot_posterior(plot_index, simulation_results, AX=None, **line_args):
    '''
    Plots the simulation results.

    Parameters
    ----------
    plot_index : dict
        A dict in the form {<model_key>: <states>} where a state is a column name 
        in the appropriate table or a key name in the appropriate obj_vals. 
    simulation_results : dict
        A dict nested as model_key -> scenario -> estimate -> (table, obj_vals)  
        where table is a DataFrame of the time response and obj_vals is a dict 
        containing the return values of the objective functions.
    AX : dict, optional
        A dict in the form {<model_key>: {<state>: <matplotlib Axes>}}. Tells the 
        function where to plost the results. If this argument is None, the Axes 
        objects are generated automatically.
        The default is None.
    **line_args : dict
        Keyword arguments for controlling the appearance of the 2D-line objects 
        to be generated. str and numerical values are applied to every line. If 
        you want to create different appeaarnces according to model and scenario, 
        use a dictionary for the value in the form {<model_key>: {<scenario>: value}}

    Returns
    -------
    figs : Figure
        Figure objects generated (if any).
    AX : dict
        A dict in the form {<model_key>: {<state>: <matplotlib Axes>}}.
    
    See Also
    --------
    simulation.plot_simulation_results
    '''
    figs, AX1 = (None, AX) if AX else utp.make_AX(plot_index)
    
    defaults   = {'label': 'scenario'}
    line_args1 = {**defaults, **line_args}
    
    sim.plot_simulation_results(plot_index, simulation_results, AX=AX1, **line_args1)
    
    return figs, AX1

def plot_exp_data(plot_index, exp_data, AX=None, **line_args):
    '''
    Plots experimental data.    

    Parameters
    ----------
    plot_index : dict
        A dict in the form {<model_key>: <states>} where a state is a column name 
        in the appropriate table or a key name in the appropriate obj_vals. 
    exp_data : dict
        A dict in the form {<model_key>: dataset} where each dataset is in turn 
        a dict in the form {(<state>, <scenario>, <'Time' or 'Data'>): <values>} 
        where state is a state in the corresponding model. Every data measurement 
        must have a corresponing time measurement. For example, ('x', 0, 'Data') 
        must be accompanied by ('x', 0, 'Time').
    AX : dict, optional
        A dict in the form {<model_key>: {<state>: <matplotlib Axes>}}. Tells the 
        function where to plost the results. If this argument is None, the Axes 
        objects are generated automatically.
        The default is None.
    **line_args : dict
        Keyword arguments for controlling the appearance of the 2D-line objects 
        to be generated. str and numerical values are applied to every line. If 
        you want to create different appeaarnces according to model and scenario, 
        use a dictionary for the value in the form {<model_key>: {<scenario>: value}}

    Returns
    -------
    figs : Figure
        Figure objects generated (if any).
    AX : dict
        A dict in the form {<model_key>: {<state>: <matplotlib Axes>}}.

    '''
    figs, AX1 = (None, AX) if AX else utp.make_AX(plot_index)

    for model_key, variables in plot_index.items():
        exp_data_model = exp_data[model_key]
        model_AX       = AX1[model_key]
        plot_exp_data_model(variables, exp_data_model, model_AX, model_key, **line_args)
        
    return figs, AX1
    
###############################################################################
#Integration with Posterior
###############################################################################  
def integrate_posteriors(sim_args, posterior):
    '''
    Performs integration while allowing the posterior to override the param_vals 
    stored in each model.

    Parameters
    ----------
    sim_args : dict
        A dict in the form: 
        {<model_key>: {'model'     : <Model>, 
                       'objectives': <objectives>}
        }
        where <objectives> is a dict of <objective_name>: <objective_function> 
        pairs.
    posterior : pandas.DataFrame
        A DataFrame of parameter values.

    Returns
    -------
    simulation_results : dict
        A dict nested as model_key -> scenario -> estimate -> (table, obj_vals). 
        Where table is a DataFrame of the time response and obj_vals is a dict 
        containing the return values of the objective functions.
    
    See Also
    --------
    simulation.integrate_models
    '''
    simulation_results = {}
    
    for model_key, value in sim_args.items():
        model      = value['model']
        posterior_ = posterior[list(model.params)] 
        _params    = dict(zip(posterior_.index, posterior_.values))
        _init      = dict(zip(model.init_vals.index,  model.init_vals.values))
        _inputs    = {i: df.values for i, df in model.input_vals.groupby(level=0)}
        
        simulation_results[model_key] = sim.integrate_model(_params=_params, _init=_init, _inputs=_inputs, **value)
    return simulation_results
    
###############################################################################
#Supporting Functions for Plotting
###############################################################################
def plot_exp_data_model(variables, exp_data_model, model_AX, model_key, **line_args):
    '''
    Handles dataset level plotting.
    
    :meta private:
    '''
    # fixed_kwargs, var_kwargs = utp.parse_line_args({**{'marker': '+', 'linestyle': ''}, **line_args})
    
    line_args_ = {**{'marker': '+', 'linestyle': ''}, **line_args}
    
    for (variable, scenario, data_type) in exp_data_model:
        if data_type == 'Time':
            continue
        elif variable in variables:
            t_key = (variable, scenario, 'Time')
            y_key = (variable, scenario, 'Data')
            
            if t_key not in exp_data_model:
                msg = 'Time points for variable {}, scenario {} not in data for model {}, .'
                raise Exception(msg.format(variable, scenario, model_key))
            
            if y_key not in exp_data_model:
                msg = 'Measurements for variable {}, scenario {} not in data for model {}, .'
                raise Exception(msg.format(variable, scenario, model_key))
            
            r    = exp_data_model[t_key], exp_data_model[y_key]
            ax   = model_AX[variable]
            args = utp.parse_recursive(line_args_, model_key, scenario, variable) 
            
            ax.plot(*r, **args)
                
    return model_AX

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.close('all')
    
    # #Test missing param check
    # model_params = ['a', 'b']
    # other_params = ['a']
    
    # try:
    #     check_missing_params(model_params, other_params, name='guess')
    #     assert False
    # except ParameterError:
    #     assert True
    # except AssertionError as e:
    #     raise e
    # except Exception as e:
    #     raise e
    
    # try:
    #     check_missing_params(other_params, model_params, name='guess')
    #     assert False
    # except AssertionError as e:
    #     assert True
    # except Exception as e:
    #     raise e
    
    # #Test unexpected param check
    # model_params  = ['a', 'b']
    # other_params  = ['a']
    # other_params_ = ['c']
    
    # try:
    #     check_unexpected_params(model_params, other_params_, name='guess')
    #     assert False
    # except ParameterError:
    #     assert True
    # except AssertionError as e:
    #     raise e
    # except Exception as e:
    #     raise e
    
    # try:
    #     check_unexpected_params(model_params, other_params, name='guess')
    #     assert False
    # except AssertionError as e:
    #     assert True
    # except Exception as e:
    #     raise e
    
    #Test simulated annealing 
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
    
    # #Read model
    # model_data = mh.read_ini('_test/TestCurveFit_3.ini')
    # models     = {key: value['model'] for key, value in model_data.items()}
    
    # #Make guess
    # guess        = {'a' : 10, 'b': 10, 'c': 2, 'e': 10, 'f': 2}
    # step_size    = {'a': 1, 'b': 1, 'e': 2}
    
    # #SSE only
    # opt_result   = simulated_annealing(models, exp_data, guess, step_size, iterations=1000)    
    # accepted     = opt_result['accepted']
    # best_row     = accepted.iloc[np.argmax(opt_result['values'])].values
    # objective    = opt_result['posterior']
    # assert np.isclose(objective(best_row), 0, atol=2)
    
    # #With priors
    # priors       = {'a': np.array([8, 0.1]), 'b': np.array([8, 0.1])}
    # opt_result   = simulated_annealing(models, exp_data, guess, step_size, iterations=1000, priors=priors)    
    # accepted     = opt_result['accepted']
    # best_row     = accepted.iloc[np.argmax(opt_result['values'])].values
    # objective    = opt_result['posterior']
    # assert all(np.isclose(best_row[:2], 8, atol=0.5))
    
    # #With priors and bounds
    # bounds       = {'a': np.array([8.5, 12]), 'b': np.array([8.5, 12])}
    # priors       = {'a': np.array([8, 0.1]),  'b': np.array([8, 0.1]) }
    # opt_result   = simulated_annealing(models, exp_data, guess, step_size, iterations=1000, priors=priors, bounds=bounds)    
    # accepted     = opt_result['accepted']
    # best_row     = accepted.iloc[np.argmax(opt_result['values'])].values
    # objective    = opt_result['posterior']
    # assert all(np.isclose(best_row[:2], 8.5, atol=0.5))
    # assert all(accepted['a'] >= 8.5)
    # assert all(accepted['b'] >= 8.5)
    
    #Test reading from .ini
    #Read model
    model_data = mh.read_ini('_test/TestCurveFit_4.ini')
    
    # #Test argument extraction
    # guesses, cf_args = get_sa_args(model_data)
    
    # models     = cf_args['models']
    # priors     = cf_args['priors']
    # bounds     = cf_args['bounds']
    # iterations = cf_args['iterations']
    # step_size  = cf_args['step_size']
    
    # assert priors['a']    == [8,   0.1]
    # assert priors['b']    == [8,   0.1]
    # assert bounds['a']    == [8.5, 12 ]
    # assert bounds['b']    == [8.5, 12 ]
    # assert bounds['e']    == [1,   12 ]
    # assert iterations     == 1000
    # assert step_size['a'] == 1
    # assert step_size['b'] == 1
    # assert step_size['e'] == 2
    # assert all([key in guesses[0] for key in ['a', 'b', 'c', 'e', 'f']])
    
    # cf_args['exp_data'] = exp_data
    
    # #Direct insertion of arguments
    # opt_result   = simulated_annealing(guess=guesses[0], **cf_args)    
    # accepted     = opt_result['accepted']
    # best_row     = accepted.iloc[np.argmax(opt_result['values'])].values
    # objective    = opt_result['posterior']
    # assert all(np.isclose(best_row[:2], 8.5, atol=0.3))
    # assert all(accepted['a'] >= 8.5)
    # assert all(accepted['b'] >= 8.5)
    
    # #Visual check
    # import matplotlib.pyplot as plt
    # plt.close('all')
    # fig = plt.figure()
    # ax  = fig.add_subplot(1, 1, 1)
    # ax.plot(accepted['a'], accepted['b'], marker='o', markersize=4)
    
    #Test plotting
    #Test plotting posterior
    posterior = pd.DataFrame.from_dict({'a': [0.5, 1],
                                        'b': [0.1, 1],
                                        'c': [2,   2],
                                        'e': [1,   1],
                                        'f': [2,   2]
                                        })
    
    plot_index = {'model_1': ['x', 'w'],
                  'model_2': ['x', 'w'],
                  }
    
    colors   = {'model_1': {0 : colors['cobalt'],
                            1 : colors['coral'],
                            },
                'model_2': colors['marigold']
                } 
    
    sim_args         = sim.get_sim_args(model_data)
    guesses, cf_args = get_sa_args(model_data)
    
    # #Test plotting posterior
    # simulation_results = integrate_posteriors(sim_args, posterior)
    # figs, AX           = plot_posterior(plot_index, simulation_results)
    
    # #Test plotting exp data
    # figs, AX = plot_exp_data(plot_index, exp_data)
    
    #Test high-level
    figs, AX, _, _ = integrate_and_plot(plot_index = plot_index, 
                                        sim_args   = sim_args, 
                                        posterior  = posterior, 
                                        guesses    = guesses, 
                                        exp_data   = exp_data,
                                        color      = colors
                                        )
    