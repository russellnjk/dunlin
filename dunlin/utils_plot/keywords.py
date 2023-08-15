from typing import Callable

import dunlin.utils as ut

def process_kwargs(kwargs    : dict|None, 
                   keys      : list[str], 
                   default   : dict|None = None, 
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
        Default values which will be merged with the keyword arguments BEFORE 
        flattening. The default is None.
    sub_args : dict, optional
        The arguments for substitution after flattening. 
    converters : dict[str, Callable]
        A dictionary that (optionally) maps the keywords in kwargs to a function. 
        The flattened value under each key will be passed to its associated 
        function using the form `new_value = function(value)`. `new_value` will
        then replace the current value.
        
        For example, a flattened argument could be `{'size': 3, 'color': 'cobalt'}`.
        Unfortunately, matplotlib's plotting functions might not recognize 
        cobalt as a valid input. In this case, we can supply a function that converts 
        `"cobalt"` to a hex value that is accepted by matplotlib's function.
        
        The default is None. More details are found in the notes.
        
    Notes
    -------
    The function can be divided into three steps:
        1. Merging `default` and `kwargs`.
        2. Flattening `kwargs`
        3. Post-processing `kwargs`.
    
        1. Merging `default` and `kwargs`
        `kwargs` and `default` are dictionaries of keyword arguments but `kwargs` 
        is provided by the end-user while `default` is implemented by the developer
        e.g. class attributes or internally-defined variables.
    
        When overlaps occur, `kwargs` will override `default`.
    
        2. Flattening `kwargs`
        `kwargs` is flattened by recursively going through each level and searching 
        for the key associated with that level. Take for example the dictionary below:
            kwargs = {'color': {'case0' : 'red',
                                'case1' : {'subcase0': 'blue',
                                           'subcase1': 'coral'
                                           }
                                }
                      }
        
        If `keys = ['case1', 'subcase1']`, the final value indexed by `'color'` 
        will be `'coral'` i.e. `{'color': 'coral'}`. If `keys = ['case0']`, 
        the value will be `'red'`. 
        
        The flattening process stops when the algorithms runs out of keys to 
        recurse through OR when it encounters a value that is not a dictionary 
        i.e. further recursion is not possible. This means that if 
        `keys = ['case0', 'subcase3']`, the value for `'color'` will still be 
        `'red'`. This is because `'red'` is a string and further recursion is 
        not possible so the algorithm does not search for `'subcase3'`. On the 
        other hand, if `keys = ['case1']`, then the the value for `'color'` 
        will be `{'subcase0': 'blue', 'subcase1': 'coral'}.`
        
        If the key cannot be found, the algorithm tries to look for the key 
        `'_default'` instead. Thus `'_default'` can be used as a catch-all. If 
        `'_default'` is not provided, then `None` will be assigned instead.
        
        3. Post-processing `kwargs`
        The values in the flattened keyword arguments will be further modified if 
            1. A function is provided by the `converters` argument
            The new value is given by `new_value = function(value)`. Allows 
            conversion of values not accepted by downstream functions into ones 
            that are. `converters` is meant to be defined only in the back-end 
            and should not be modifiable to end-users.
            
            2. The value is itself a callable
            The new value is given by `new_value = value(**sub_args)`. `sub_args`
            is meant to be defined only in the back-end and should not be modifiable 
            to end-users. It will contain information relevant to the model and 
            plotting such as the model's `ref` attribute. The user can define a 
            function that provides a highly customized value based on sub_args.
            
            This is useful for things like labels for which a user may have a 
            preferred format.
            
            3. The value has the attribute `format`
            The new value is given by `new_value = value.format(**sub_args)`. 
            Similar to the second case.
        
        Both `converters` and `sub_args` are meant to be defined by in the 
        back-end and should not be modifiable to end-users.
        
    Returns
    -------
    kwargs
        A dictionary of keyword arguments suitable for passing into downstream 
        plotting functions.

    '''
    
    #Merge default and kwargs
    #Kwargs always overwrites default in the event of an overlap
    if default and kwargs:
        kwargs = {**default, **kwargs}
    elif default:
        kwargs = default
    elif kwargs:
        pass
    else:
        kwargs = {}
    
    #Flatten the kwargs if keys are provided
    if keys:
        kwargs = flatten_kwargs(kwargs, keys)
    
    #Carry out post-processing
    sub_args = {} if sub_args is None else sub_args
    
    for key in kwargs:
        old_value = kwargs[key]
        converter = converters.get(key)
        
        if converter:
            new_value = converter(old_value)
        elif callable(old_value):
            new_value = old_value(**sub_args)
        elif hasattr(old_value, 'format'):
            new_value = old_value.format(**sub_args)
        
        else:
            new_value = old_value
        
        kwargs[key] = new_value
            
    return kwargs
    
###############################################################################
#Flattening Keyword Args
###############################################################################
def flatten_kwargs(dct: dict, keys: list) -> dict:
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
    
    if type(result) == dict:
        if len(keys) == 1:
            raise ValueError(f'The dictionary is too deeply nested. Check the number of levels: {dct}')
        return recursive_get(result, *keys[1:])
    else:
        return result
