import numpy  as np
import pandas as pd
from numba import njit

def code2func(codes):
    g = globals()
    
    if type(codes) in [list, tuple]:
        func_name, code = codes
        exec(code, g)
        func      = g[func_name]
        func.code = code
        return func
    
    else:
        funcs = {}
        g     = globals()
        for key, (func_name, code) in codes.items():
            
            exec(code, g)
            func       = g[func_name]
            func.code  = code
            funcs[key] = func
        return funcs  
