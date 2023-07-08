from datetime  import datetime
from numbers   import Number
from pyrfc3339 import generate
from typing    import Callable, Union

from . import readprimitive as rpr

Primitive = Union[Number, str, bool, datetime]
###############################################################################
#Writing Elements
###############################################################################
def write_list(lst: list) -> str:
    chunks = []
    #Iterate and extend
    for x in lst:
        if type(x) == dict:
            chunk = write_dict(x, multiline_dict=False) 
            chunk = '[' + chunk + ']'
        elif type(x) in [list, tuple]:
            chunk = write_list(x)
        else:
            chunk = write_primitive(x)
        
        chunks.append(chunk)
        
    code = '[' + ', '.join(chunks) + ']'
    return code

def write_dict(dct            : dict, 
               multiline_dict : bool     = True, 
               n_format       : Callable = str,
               _level         : int      = 0
               ) -> str:
    chunks = []
    
    for key, value in dct.items():
        #Convert the key
        if type(key) in [list, tuple]:
            key_ = write_list(key)
        else:
            key_ = write_primitive(key, n_format)
        
        #Convert the value
        if type(value) == dict:
            value_ = write_dict(value, 
                                multiline_dict = multiline_dict, 
                                _level         = _level+1
                                )
        
        elif type(value) in [list, tuple]:
            value_ = write_list(value)
        
        else:
            value_ = write_primitive(value)
        
        chunk = f'{key_} : {value_}'
        chunks.append(chunk)
    
    if _level == 0:
        if multiline_dict:
            code = '\n'.join(chunks)
        else:
            code = ', '.join(chunks)
    else:
        if multiline_dict:
            indent = '\t'*_level
            code   = f'[\n{indent}' + f',\n{indent}'.join(chunks) + f'\n{indent}],'
        
        else:
            code = '[' + ', '.join(chunks) + ']'
    
    return code
        

###############################################################################
#Writing Primitives
###############################################################################
def write_primitive(x: Primitive, n_format: Callable=str) -> str:
    if isinstance(x, Number):
        return n_format(x)
    elif type(x) == str:
        if needs_quotes(x):
            return repr(x)
        else:
            return x
    elif type(x) == bool:
        return str(x)
    elif type(x) == datetime:
        return repr(generate(x, accept_naive=True, microseconds=True))
    else:
        raise TypeError(f'Unexpected type: {type(x).__name__}')
    
special_characters = set('!$#`;:,')
def needs_quotes(string: str) -> bool:
    if special_characters.intersection(string):
        return True
    
    if len(string) >= 2:
        return string[0] in '\'"' and string[0] != string[-1]
    try:
        r = rpr.read_primitive(string) != string
    except:
        return True
    
    return r