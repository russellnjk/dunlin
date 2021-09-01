import numpy  as np
import pandas as pd
from numba import njit

_g = globals()

def code2func(codes):
    
    if type(codes) in [list, tuple]:
        func_name, code = codes
        exec(code, _g)
        func      = _g[func_name]
        func.code = code
        return func
    
    else:
        funcs = {}
        for key, (func_name, code) in codes.items():
            
            exec(code, _g)
            func       = _g[func_name]
            func.code  = code
            funcs[key] = func
        return funcs  
