import numpy       as     np
import pandas      as     pd
from   numba       import jit
from   scipy.stats import norm

###############################################################################
#Optimizers
###############################################################################
def simulated_annealing(func, guess, transition, check_bounds=None, iterations=10000, SA=True):
    '''
    Finds the parameters that MAXIMIZE an objective function.

    Parameters
    ----------
    func : function
        The function to be maximized that accepts a Numpy array and returns a 
        single numerical value.
    guess : dict
        A dictionary in the form {<param_name>: <param_value>}.
    transition : function
        A function that accepts a Numpy array and returns a new array.
    check_bounds : function, optional
        A function for checking the validity of a set of parameters that accepts 
        a Numpy array and returns a boolean. The parameter set is rejected if 
        the function returns False and vice versa. The default is None.
    iterations : int, optional
        The number of iterations to take. The default is 10000.
    SA : bool, optional
        If True, the temperature parameter of the acceptance criterion changes 
        with each step. If False, the temperature remains constant and the algorithm
        becomes a typial Gibbs sampler. The default is True.

    Returns
    -------
    optimization_result: dict
        A dict in the form {'accepted': accepted, 'values': posterior}.

    '''
    curr_params        = np.array(list(guess.values()))
    next_params        = None
    curr_log_posterior = func(curr_params)
    accept             = True
    accepted           = [curr_params]
    posterior          = [curr_log_posterior]
    
    for i in range(iterations):
        
        next_params = transition(curr_params)
        
        if check_bounds is not None:   
            if not check_bounds(next_params):
                continue
                
        next_log_posterior = func(next_params)
        
        temp   = (1-1*i/iterations) if SA else 1
        accept = acceptance_criterion(next_log_posterior, curr_log_posterior, temp)
        
        #Collect sample
        if accept:
            curr_params        = next_params
            curr_log_posterior = next_log_posterior
            accepted.append(next_params)
            posterior.append(next_log_posterior)
            
    return {'accepted': accepted, 'values': posterior}

@jit(nopython=True)
def get_exp_substract(a, b):
    '''
    :meta private:
    '''
    return np.exp(a-b)

@jit(nopython=True)
def acceptance_criterion(next_log_posterior, curr_log_posterior, temp):
    '''
    :meta private:
    '''
    test     = np.random.rand()
    p_accept = np.exp(next_log_posterior - curr_log_posterior)
    return p_accept > test**temp

###############################################################################
#Transition
###############################################################################
def wrap_transition(step_size, param_index, n_iterations=10000):
    '''
    Returns a function for performing the transition step in simulated annealing.

    Parameters
    ----------
    step_size : dict
        A dict in the form {<param_name>: <std_dev>}.
    param_index : dict
        DESCRIPTION.
    n_iterations : int, optional
        The number of iterations. The default is None.

    Returns
    -------
    transition: function
        A function that accepts a Numpy array and returns a new array.
    '''
    pairs   = []
    for param, value in step_size.items():
        if value == 0:
            continue
        
        try:
            index = param_index[param]
        except:
            msg = 'Could not find {} in param_names.'
            raise KeyError(msg.format(param))
        
        try:
            index = int(index)
        except:
            msg = 'Could not convert the value indexed at {} in param_names into an int: {}'
            raise TypeError(msg.format(param, index))
        
        pair = [value, index]
        pairs.append(pair)
    
    pairs   = np.array(sorted(pairs, key=lambda pair: pair[1]))
    indices = pairs[:,1].astype(np.int64)
    stepper = norm(loc=0, scale=pairs[:,0]).rvs
    
    def transition(params_array):
        delta            = stepper()
        new_params_array = copy_add(params_array, indices, delta)

        return new_params_array
    return transition

@jit(nopython=True)
def copy_add(params_array, indices, delta):
    '''
    :meta private:
    '''
    new_params_array           = params_array.copy().astype(np.float64)
    new_params_array[indices] += delta
    return new_params_array
    
###############################################################################
#Log-Posterior Calculation
###############################################################################
def wrap_get_log_posterior(get_log_likelihood, get_log_prior=None, callback=None):
    '''
    Returns a function for calculating the log-posterior of a parameter set.

    Parameters
    ----------
    get_log_likelihood : function
        A function for calculating the log-likelihood a parameter set that accepts 
        a Numpy array and returns a single numerical value.
    get_log_prior: function, optional
        A function for calculating the log-prior of a parameter set that accepts 
        a Numpy array and returns a single numerical value. The default is None.
    callback : function or tuple of functions, optional
        A function(s) to be called using func(log_posterior, log_prior, log_likelihood). 
        Keywords are not to be used. The default is None.

    Returns
    -------
    get_log_posterior: function
        A function to be maximized that accepts a Numpy array and returns a 
        single numerical value.
    '''

    if not callable(get_log_likelihood):
        msg = 'get_log_likelihood must be a function. Received: {}'
        raise TypeError(msg.format(get_log_likelihood))
        
    callbacks     = []
    if callback:
        if callable(callback):
            callbacks = [callback]
        elif type(callback) in [list, tuple]:
            callbacks = tuple(callbacks)
        elif type(callback) == dict:
            callbacks = tuple(callbacks.values())
        else:
            msg = 'callback must be a function, a list of functions or a dict of indexed functions. Received {} instead.'
            raise TypeError(msg)
        
        if not all([callable(func) for func in callbacks]):
            msg = 'callback must be a function, a list of functions or a dict of indexed functions. Detected non-function objects.'
            raise TypeError(msg)
            
    def get_log_posterior(params_array):
        log_prior      = get_log_prior(params_array) if get_log_prior else 0
        log_likelihood = get_log_likelihood(params_array) 
        log_posterior  = log_prior + log_likelihood
        
        if callbacks:
            try:
                [func(log_posterior, log_prior, log_likelihood) for func in callbacks]
            except Exception as e:
                args   = e.args
                msg    = '\n'.join( ('An error occured in trying to evaluate a callback function.',) )
                e.args = (msg,)
                raise e
                
        return log_posterior
    
    return get_log_posterior
    
###############################################################################
#Log-Prior Calculation
###############################################################################
def wrap_get_log_prior(priors_dict, param_index):
    '''
    Returns a function for calculating the log-prior of a parameter set.

    Parameters
    ----------
    priors_dict : dict
        A dict of the form {<param_name>: [<mean>, <std_dev>]}. You do not need 
        to provide a prior for every parameter.
    param_index : dict
        A dict in the form {<param_name>: <index>}. The index indicates the order 
        of the parameters and must start from 0.

    Returns
    -------
    get_log_prior: function
        A function for calculating the log-prior of a parameter set that accepts 
        a Numpy array and returns a single numerical value.
    '''
    #Check priors
    if priors_dict is None:
        return None
    elif type(priors_dict) == dict:
        if len(priors_dict) == 0:
            return None
        
        temp = [param for param in priors_dict if param not in param_index]
        if temp:
            msg = 'Detected unexpected keys in priors: {}'
            raise ValueError(msg.format(temp))
        
        temp    = [[value[0], value[1], param_index[param]] for param, value in priors_dict.items()]
        temp    = np.array(sorted(temp, key=lambda x: x[2]))
        indices = temp[:,2].astype(np.int64)
        priors  = norm(loc=temp[:,0], scale=temp[:,1]).pdf
        
    else:
        msg = 'priors must be a dict. Received {} instead.'
        raise TypeError(msg.format(type(priors_dict)))
        
    def get_log_prior(params_array):
        prior_values = priors(params_array[indices])
        return log_sum(prior_values)
    
    return get_log_prior

@jit(nopython=True)
def log_sum(a):
    '''
    :meta private:
    '''
    return np.log(a).sum()

###############################################################################
#Bounds Checking
###############################################################################
def wrap_check_bounds(bounds_dict, param_index):
    '''
    Returns a function for checking the validity of a parameter set.

    Parameters
    ----------
    bounds_dict : dict
        A dict in the form {<param_name>: [<lower>, <upper>]}. You do not need 
        to provide bounds for every single parameter.
    param_index : dict
        A dict in the form {<param_name>: <index>}. The index indicates the order 
        of the parameters and must start from 0.

    Returns
    -------
    check_bounds: function
        A function for checking the validity of a set of parameters that accepts 
        a Numpy array and returns a boolean. The parameter set is rejected if 
        the function returns False and vice versa.
    '''
    if not bounds_dict:
        return None
    elif type(bounds_dict) == dict:
        temp = [param for param in bounds_dict if param not in param_index]
        if temp:
            msg = 'Detected unexpected keys in priors: {}'
            raise ValueError(msg.format(temp))
        
        temp    = [[value[0], value[1], param_index[param]] for param, value in bounds_dict.items()]
        temp    = np.array(sorted(temp, key=lambda x: x[2]))
        indices = temp[:,2].astype(np.int64)
        upper   = temp[:,1]
        lower   = temp[:,0]

        if not all(upper > lower):
            raise ValueError('Detected upper bounds that were lower or equal to lower bounds.')
    else:
        msg = 'bounds must be a dict. Received {} instead.'
        raise TypeError(msg.format(type(bounds_dict)))
                
    def check_bounds(params_array):
        temp = params_array[indices] < lower
        if temp.any():
            return False
        
        temp = params_array[indices] > upper
        if temp.any():
            return False
        
        return True
    return check_bounds

if __name__ == '__main__':
    #Test prior
    priors_dict   = {'a': np.array([1, 0.1]), 'b': np.array([1, 0.05])}
    param_index   = {'a': 0, 'b': 1, 'c': 2}
    get_log_prior = wrap_get_log_prior(priors_dict, param_index)
    
    log_prior1    = get_log_prior(np.array([0.9, 0.95, 2]))
    log_prior0    = get_log_prior(np.array([  1,    1, 2]))
    assert np.isclose(log_prior1/log_prior0, 0.711019, atol=1e-3)
    
    #Test transition
    step_size        = {'a': 0.05, 'b': 0.05}
    param_index      = {'a': 0, 'b': 1, 'c': 2}
    transition       = wrap_transition(step_size, param_index)
    new_params_array = transition(np.array([0, 0, 0]))
    assert len(new_params_array) == 3
    assert new_params_array[2]   == 0
    
    #Test bounds check
    bounds_dict  = {'a': np.array([0.1, 1]), 'b': np.array([0.05, 1])}
    param_index  = {'a': 0, 'b': 1, 'c': 2}
    check_bounds = wrap_check_bounds(bounds_dict, param_index)
    in_bounds    = check_bounds(np.array([0.5, 0.5, 0.5]))
    assert in_bounds == False
    
    bounds_dict  = {'a': np.array([0.1, 1]), 'b': np.array([0.05, 1])}
    param_index  = {'a': 0, 'b': 1, 'c': 2}
    check_bounds = wrap_check_bounds(bounds_dict, param_index)
    in_bounds    = check_bounds(np.array([0.5, 0, 0.5]))
    assert in_bounds == True
    
    #Test posterior
    param_index        = {'a': 0, 'b': 1, 'c': 2}
    get_log_likelihood = lambda params: np.sum(-(params - 1)**2)
    get_log_posterior  = wrap_get_log_posterior(get_log_likelihood)
    r                  = get_log_posterior(np.array([1, 1, 1]))
    assert r == 0
    
    get_log_posterior  = wrap_get_log_posterior(get_log_likelihood)
    r                  = get_log_posterior(np.array([0, 1, 2]))
    assert r == -2
    
    get_log_posterior  = wrap_get_log_posterior(get_log_likelihood)
    r                  = get_log_posterior(np.array([2, 1, 2]))
    assert r == -2
    
    get_log_posterior  = wrap_get_log_posterior(get_log_likelihood)
    r                  = get_log_posterior(np.array([1, 1, 3]))
    assert r == -4 
    
    #Test simulated annealing
    guess       = {'a': 2, 'b': 2, 'c': 0}
    priors_dict = {'a': np.array([1, 0.1]), 'b': np.array([1, 0.05])}
    bounds_dict = {'a': np.array([1, 0.1]), 'b': np.array([1, 0.05])}
    step_size   = {'a': 0.05, 'b': 0.05}
    param_index = {'a': 0, 'b': 1, 'c': 2}
    
    transition         = wrap_transition(step_size, param_index)
    check_bounds       = wrap_check_bounds(bounds_dict, param_index)
    get_log_prior      = wrap_get_log_prior(priors_dict, param_index)
    get_log_likelihood = lambda params: np.sum(-(params - 1)**2)
    get_log_posterior  = wrap_get_log_posterior(get_log_prior, get_log_likelihood)
    
    result   = simulated_annealing(get_log_posterior, guess, transition, check_bounds, iterations = 500)
    accepted = result['accepted']
    adf      = pd.DataFrame(accepted, columns=list(param_index.keys()) )
    assert all(adf['c'] == 0)
    assert all(np.isclose(adf.iloc[-1].values, [1, 1, 0], atol=1e-1))
    
    # #Visual check
    # import matplotlib.pyplot as plt
    # plt.close('all')
    # fig = plt.figure()
    # ax  = fig.add_subplot(1, 1, 1)
    # ax.plot(adf['a'], adf['b'], marker='o', markersize=4)    

    