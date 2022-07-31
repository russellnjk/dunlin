import numpy as np
from numba import njit
from time import time

@njit
def f(x, y):
    
    if y > 0:
        return 0
    
    z = 0
    for i in range(10**7):
        z += np.sum(x-y)**.5
        
    return z


if __name__ == '__main__':
    x  = np.array([1, 2, 3])
    xx = np.linspace(0, 100, 1001)
    
    def timeit(x, y):
        start = time()
        f(x, y)
        stop  = time()
        
        print('{:.4f}'.format(stop-start))
    
    timeit(x, -1)
    # timeit(1)
    timeit(xx, -1)