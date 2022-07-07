import numpy        as np
import numpy.random as rnd
import scipy.optimize as sop

from  collections import namedtuple
from  numba       import njit
from  tqdm        import tqdm

Result = namedtuple('Result', 'samples posterior context other', defaults=[None])

###############################################################################
#Local Minimize
###############################################################################
def local_minimize(func, x0, bounds, callback=None, **kwargs):
    samples   = []
    posterior = []
    states    = []
    
    def _callback_trust_constr(xk, state):
        states.append(state)
        posterior.append(func(xk))
        samples.append(xk)
        
        if callable(callback):
            return callback(xk, state)
        else:
            return 
    
    def _callback_general(xk):
        states.append(True)
        posterior.append(func(xk))
        samples.append(xk)
        
        if callable(callback):
            return callback(xk)
        else:
            return 
    
    method = kwargs.get('method', None)
    if method == 'trust-constr':
        _callback = _callback_trust_constr  
    else: 
        _callback_general
    
    result = sop.minimize(func, x0, callback=_callback, **kwargs)
    
    return Result(samples, posterior, states, result)    

###############################################################################
#Differential Evolution
###############################################################################
def differential_evolution(func, bounds, callback=None, **kwargs):
    samples      = []
    posterior    = []
    convergences = []
    
    def _callback(xk, convergence):
        convergences.append(convergence)
        posterior.append(func(xk))
        samples.append(xk)
        
        if callable(callback):
            return callback(xk, convergence)
        else:
            return 
    
    result = sop.differential_evolution(func, bounds, callback=_callback, **kwargs)
    
    return Result(samples, posterior, convergences, result)

###############################################################################
#Dual Annealing
###############################################################################
def dual_annealing(func, bounds, x0, callback=None, **kwargs):
    samples   = []
    posterior = []
    contexts  = []
    
    def _callback(x, f, context):
        contexts.append(context)
        posterior.append(f)
        samples.append(x)
        
        if callable(callback):
            return callback(x, f, context)
        else:
            return 

    result = sop.dual_annealing(func, bounds, callback=_callback, **kwargs)
    
    return Result(samples, posterior, contexts, result)

###############################################################################
#Basinhopping
###############################################################################
def basinhopping(func, bounds, x0, callback=None, accept_test=None, **kwargs):
    samples   = []
    posterior = []
    accepted  = []
    
    def _callback(x, f, accept):
        accepted.append(accept)
        posterior.append(f)
        samples.append(x)
        
        if callable(callback):
            return callback(x, f, accept)
        else:
            return 
    
    def _accept_test(f_new, x_new, f_old, x_old):
        if not bounds(x_new=x_new):
            return False
        elif callable(accept_test):
            return accept_test(f_new=f_new, x_new=x_new, f_old=f_old, x_old=x_old)
        else:
            return True
    
    
    result = sop.basinhopping(func, 
                              x0, 
                              accept_test=_accept_test, 
                              callback=_callback,
                              **kwargs
                              )
    
    return Result(samples, posterior, accepted, result)
        
###############################################################################
#Simulated Annealing
###############################################################################
def simulated_annealing(func, bounds, step, x0, niter=1000, cooling=1, disp=False):
    x_curr    = np.array(x0)
    pos_curr  = func(x_curr)
    init      = f"{pos_curr:.4}"
    min_val   = pos_curr
    samples   = [x_curr]
    posterior = [pos_curr]
    accepted  = [True]
    
    #Set up the loop variables
    pbar = range(niter)
    if disp:
        pbar = tqdm(pbar, position=0, leave=True)
        msg  = f'{init} -> {init}'
        pbar.set_description(msg)

    #Iterate
    for i in pbar:
        x_new = step(x_curr)
        
        if not bounds(x_new=x_new):
            continue
        
        pos_new = func(x_new)
        accept  = calculate_inverted_acceptance(pos_curr, pos_new, i, niter, cooling)
        
        samples.append(x_new)
        posterior.append(pos_new)
        accepted.append(accept)
        
        #Update the the objective value in the display
        if disp and pos_new < min_val:
            msg = f'{init} -> {pos_new:.4}'
            pbar.set_description(msg)
            min_val = pos_new
        
        #Reassign and prepare for next iteration
        if accept:
            x_curr   = x_new
            pos_curr = pos_new

        if disp:
            pbar.update(1)

    if disp:
        pbar.close()
    
    return Result(samples, posterior, accepted)

@njit
def calculate_inverted_acceptance(pos_curr, pos_new, i, niter, cooling):
    #IMPT IMPT IMPT IMPT IMPT IMPT IMPT IMPT IMPT IMPT 
    #Inverted due to minimization instead of maximization
    criterion = np.exp(pos_curr - pos_new)
    test      = rnd.uniform(0, 1) 
    test      = test*np.exp(-i/niter*cooling)
    
    return criterion > test
    
        
    