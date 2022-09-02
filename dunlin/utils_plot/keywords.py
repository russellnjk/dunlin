from typing import Callable

import dunlin.utils as ut

def process_kwargs(kwargs: dict, 
                   keys: list[str], 
                   default: dict=None, 
                   sub_args: dict=None, 
                   converters: dict[Callable]=None
                   ) -> dict:
    
    '''
    The master function for processing keyword arguments for plotting.                   

    Parameters
    ----------
    kwargs : dict
        The user-supplied values for processing.
    keys : list
        The keys for recursive flattening.
    default : dict, optional
        Default values which will be merged with the flattened keyword arguments. 
        The default is None.
    sub_args : dict, optional
        The arguments for substitution after flattening. Supposed to supplied 
        from the backend. The default is None.
    converters : dict[str, Callable]
        Pairs of keywords and callables. The keyword indicates which keyword 
        argument requires substitution. The callables (or None) provides an 
        additional layer of parsing for that keyword argument. The default is None.

    Returns
    -------
    kwargs
        A dictionary of keyword arguments suitable for passing into external 
        plotting functions.

    '''
    if keys:
        kwargs = flatten_kwargs(kwargs, keys)
    else:
        kwargs = dict(kwargs)
        
    if default:
        kwargs = {**default, **kwargs}
        
    for key in kwargs:
        converter = converters.get(key)
        substitute(kwargs, key, converter, sub_args)
    
    return kwargs
    
###############################################################################
#Flattening Keyword Args
###############################################################################
def flatten_kwargs(dct, keys):
    if dct is None:
        flattened = {}
    else:
        flattened = {k: recursive_get(v, *keys) for k, v in dct.items()}
    
    return flattened
    
def recursive_get(dct, *keys):
    if type(dct) != dict:
        return dct
    
    result = dct.get(keys[0], None)
    
    if type(result) == dict:
        if len(keys) == 1:
            raise ValueError(f'The dictionary is too deeply nested. Check the number of levels: {dct}')
        return recursive_get(result, *keys[1:])
    else:
        return result

###############################################################################
#Parsing Flattened Args
###############################################################################
#TODO: Deprecate this
def replace(kwargs, key, default, _converter=None, **repl_args):
    value = kwargs.get(key, default)
    
    if callable(value):
        value = value(**repl_args)
    elif hasattr(value, 'format'):
        value = str(value).format(**repl_args)
        
    if _converter:
        value = _converter(value)
    
    kwargs[key] = value

#TODO: Deprecate this
def call(kwargs, _skip=('label', 'color'), _converter=None, **repl):
    for key in kwargs:
        if key in _skip:
            continue
        else:
            value = kwargs[key]
            
            if callable(value):
                value = value(**repl)
            
            if _converter:
                value = _converter(value)
            
            kwargs[key] = value

def substitute(kwargs, key, converter=None, sub_args=None):
    sub_args = {} if sub_args is None else sub_args
    value    = kwargs[key]
    
    if callable(value):
        value = value(**sub_args)
    elif hasattr(value, 'format'):
        value = value.format(**sub_args)
        
    if converter:
        value = converter(value)
    
    kwargs[key] = value
    
