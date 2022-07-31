import numpy as np
from numba import njit

@njit
def make_sparse(points, atol=0.01):
    result = [points[0]]
    for i in range(1, len(points)):
        p = points[i]

        skip = False
        for pp in result:
            #Eliminate points outside the bounding box
            if pp[0] > p[0] + atol or pp[0] < p[0] - atol or pp[1] > p[1] + atol or pp[1] < p[1] - atol:
                continue
            elif pp[1] > p[1] + atol or pp[1] < p[1] - atol:
                continue
            
            #Calculate dist for points inside
            test = np.sum( ((p-pp)/atol)**2 ) 
            
            if test < 1:
                skip = True
                break
        
        if not skip:
            result.append(p)

    return result

def warmup():
    make_sparse(np.array([ [0.1, 0.1]]), 0.1)
    
warmup()
