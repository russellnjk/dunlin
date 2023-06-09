from typing import Callable

import dunlin.utils as ut

def process_kwargs(kwargs    : dict, 
                   keys      : list[str], 
                   default   : dict=None, 
                   sub_args  : dict=None, 
                   converters: dict[Callable]=None
                   ) -> dict:
    
    '''
    The master function for flattening keyword arguments for plotting. An example 
    of an input argument could be this: 
        
        `{'size': 5, 
          'color': {'case0': 'blue', 
                    'case1': {'subcase0': 'red',
                              'subcase1': 'orange'
                              }
                    }
          }`
    
    We want the size to be 5 but the color should be blue for case0, red for 
    case1/subcase0 and orange for case1/subcase1. If the keys provided are 
    `['case0', 'subcase1']`, this function will return a dictionary where 
    `color: 'orange'`.                 

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
        The arguments for substitution after flattening. 
    converters : dict[str, Callable]
        A dictionary that maps the keywords in kwargs to a callable. The flattened 
        value under each key will be passed to its associated callable. This provides 
        an additional layer of processing and is useful when the flattened value 
        is not a valid argument to downstream functions. 
        
        For example, a flattened argument could be `{'size': 3, 'color': 'cobalt'}`.
        Unfortunately, matplotlib's plotting functions might not recognize 
        cobalt as a valid input. In this case, we can supply a function that converts 
        `"cobalt"` to a tuple of RGB values that are accepted by matplotlib's 
        function. 
        
        Callables only need to be provided for arguments that expected to require 
        additional processing. The default is None.

    Returns
    -------
    kwargs
        A dictionary of keyword arguments suitable for passing into downstream 
        plotting functions.

    '''
    if keys:
        kwargs = flatten_kwargs(kwargs, keys)
    elif not kwargs:
        kwargs = {}
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
    
    #Modified Jun 2023
    default = dct.get('_default', None)
    result  = dct.get(keys[0], default)
    # try:
    #     result = dct[keys[0]]
    # except KeyError:
    #     if keys[0] not in dct and '_default' in dct:
    #         result = dct['_default']
    #     else:
    #         result = None
    # result = dct.get(keys[0])
    
    if type(result) == dict:
        if len(keys) == 1:
            raise ValueError(f'The dictionary is too deeply nested. Check the number of levels: {dct}')
        return recursive_get(result, *keys[1:])
    else:
        return result

###############################################################################
#Parsing Flattened Args
###############################################################################

#Commented out Jun 2023. Delete this if everything has been working since then.
# #TODO: Deprecate this
# def replace(kwargs, key, default, _converter=None, **repl_args):
#     value = kwargs.get(key, default)
    
#     if callable(value):
#         value = value(**repl_args)
#     elif hasattr(value, 'format'):
#         value = str(value).format(**repl_args)
        
#     if _converter:
#         value = _converter(value)
    
#     kwargs[key] = value

# #TODO: Deprecate this
# def call(kwargs, _skip=('label', 'color'), _converter=None, **repl):
#     for key in kwargs:
#         if key in _skip:
#             continue
#         else:
#             value = kwargs[key]
            
#             if callable(value):
#                 value = value(**repl)
            
#             if _converter:
#                 value = _converter(value)
            
#             kwargs[key] = value

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
    
