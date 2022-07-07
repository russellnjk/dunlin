import dunlin.utils as ut

###############################################################################
#Flattening Keyword Args
###############################################################################
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
def replace(kwargs, key, default, _converter=None, **repl_args):
    value = kwargs.get(key, default)
    
    if callable(value):
        value = value(**repl_args)
    elif value is None:
        pass
    else:
        value = str(value).format(**repl_args)
    
    if _converter:
        value = _converter(value)
    
    kwargs[key] = value

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