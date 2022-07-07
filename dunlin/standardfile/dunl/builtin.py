import numpy as np

def linspace(start, stop, npoints, *extras):
    array = np.linspace(start, stop, npoints)
    lst   = list(array) + list(extras)
    return list(np.unique(lst))


builtin_functions = {'linspace': linspace}

