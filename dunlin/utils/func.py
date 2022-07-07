import numpy  as np
import pandas as pd
from numba import njit

default_globals = {'__np' : np, '__njit': njit}

def code2func(code, *func_names, globalvars=None):
    __locals  = {}
    __globals = default_globals if globalvars is None else globalvars
    
    exec(code, 
         __globals, 
         __locals
         )
    
    result = {i: __locals[i] for i in func_names}
    
    return result