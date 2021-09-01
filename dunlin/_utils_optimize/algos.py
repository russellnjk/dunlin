import numpy        as np
import numpy.random as rnd
from   numba import njit
from   tqdm  import tqdm

def simulated_annealing(func, bounds, step, x0, niter=1000, cooling=1, disp=False):
    accepted  = []
    posterior = []
    x_curr    = np.array(x0)
    pos_curr  = func(x_curr)
    to_iter   = tqdm(range(niter)) if disp else range(niter)
    for i in to_iter:
        x_new = step(x_curr)
        
        if not bounds(x_new=x_new):
            continue
        
        pos_new = func(x_new)
        accept = calculate_inverted_acceptance(pos_curr, pos_new, i, niter, cooling)
        if accept:
            accepted.append(x_new)
            posterior.append(pos_new)
            
            x_curr   = x_new
            pos_curr = pos_new
            
        
    if not len(accepted):
        raise ValueError('No accepted samples.')
    return accepted, posterior

@njit
def calculate_inverted_acceptance(pos_curr, pos_new, i, niter, cooling):
    #IMPT IMPT IMPT IMPT IMPT IMPT IMPT IMPT IMPT IMPT 
    #Inverted due to minimization instead of maximization
    criterion = np.exp(pos_curr - pos_new)
    test      = rnd.uniform(0, 1) 
    test      = test*np.exp(-i/niter*cooling)
    
    return criterion > test
    
        
    