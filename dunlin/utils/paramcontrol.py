'''
Decorators for:
    1. Aliasing params 
    2. Limiting parameter values

'''

###############################################################################
#General Decorator Error
###############################################################################
class DecoratorError(Exception):
    pass

###############################################################################
#Aliasing parameters
###############################################################################
def alias_param(**ori2alias):
    '''
    A decorator that allows aliasing of parameters. Renames aliased arguments 
    under their original parameter names before passing them into the wrapped 
    function. If used with accept_vals, it must come BEFORE.

    Parameters
    ----------
    **ori2alias : dict-like
        Pairs of original parameter names and their aliases. Keys and values must 
        be strings.

    '''
    ori2alias = ori2alias
    oris      = set(ori2alias.keys())
    alias2ori = {}
    
    for k, v in ori2alias.items():
        if v == k or not v or not k:
            raise DecoratorError(f'Invalid alias: {k} -> {v}')
        
        alias2ori[v] = k
    
    if len(alias2ori) != len(ori2alias):
        raise ValueError('Aliases and original param names must be 1-1.')
    
    def wrapper(function):
        def helper(*args, **kwargs):
            new_kwargs = {}
            for k, v in kwargs.items():
                
                if k in new_kwargs:
                    #Python doesn't allow repeated kwargs
                    #Thus this should be a repeat in param/alias pairs
                    raise ParamAliasError(k, ori2alias[k])
                    
                elif k in oris:
                    #k is in the original form
                    new_kwargs[k] = v
                
                else:
                    #k may be aliased parameter
                    ori = alias2ori.get(k)
                    
                    if ori is None:
                        #k is not alised
                        new_kwargs[k] = v
                    elif ori in new_kwargs:
                        print(new_kwargs)
                        raise ParamAliasError(k, alias2ori[k])
                    else:
                        new_kwargs[ori] = v
            
            return function(*args, **new_kwargs)
        return helper
    return wrapper

class ParamAliasError(Exception):
    def __init__(self, *names):
        s = f'Repeated arguments: {names}. You may need to check your use of parameter aliases.'
        super().__init__(s)

###############################################################################
#Limiting parameter values
###############################################################################
def accept_vals(**arg2vals):
    '''
    A decorator that limits the acceptable values for a parameter.

    Parameters
    ----------
    **arg2vals : dict-like
        Pairs of argument names and values to be tested against. The values can 
        be in a list-like format or a callable in which it works as a test function. 
        If callable, it can also have a text attribute for printing during errors.

    '''
    for test in arg2vals.values():
        if not callable(test) and not hasattr(test, '__contains__') and test is not None:
            raise DecoratorError('Test values in accept_vals must be list-like or a callable.')
    
    def raise_exception(arg_name, arg_val):
        msg = 'No implementation for parsing argument {} when its value is {}.'
        msg = msg.format(arg_name, arg_val)
        
        test = arg2vals[arg_name]
        
        if hasattr(test, 'text'):
            msg += f' Required condition: {test.text}'
        elif hasattr(test, '__iter__'):
            msg += f' Accepted values are: {test}'
        
        raise NotImplementedError(msg)
    
    def test_arg(arg_name, arg_val):
        if arg_name not in arg2vals:
            raise ValueError(f'Unexpected argument {arg_name}')
            
        test = arg2vals[arg_name]
        if test is None:
            pass
        elif callable(test):
            try:
                result = test(arg_val)
            except:
                raise_exception(arg_name, arg_val)
            if not result:
                raise_exception(arg_name, arg_val)
        else:
            if arg_val not in test:
                raise_exception(arg_name, arg_val)
        
        return
                
    def wrapper(func):
        def helper(*args, **kwargs):
            #Check args
            temp = zip(args, arg2vals.keys())
            for arg_val, arg_name in temp:
                test_arg(arg_name, arg_val)
            
            #Check kwargs
            for arg_name, arg_val in kwargs.items():
                test_arg(arg_name, arg_val)
            
            return func(*args, **kwargs)
        return helper
    return wrapper
