import numpy       as     np
import pandas      as     pd
from   numba       import jit
from   scipy.stats import norm

###############################################################################
#Transition
###############################################################################
def wrap_transition(model):
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