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
            code   = f'[\n{indent}' + f',\n{indent}'.join(chunks) + f'\n{indent}]'
        
        else:
            code = '[' + ', '.join(chunks) + ']'
    
    return code
        

###############################################################################
#Writing Primitives
###############################################################################
def write_primitive(x: Primitive, n_format: Callable=str) -> str:
    if isinstance(x, Number):
        return n_format(x)
    
    elif isinstance(x, bool):
        return str(x)
    
    elif isinstance(x, datetime):
        return repr(generate(x, accept_naive=True, microseconds=True))
    
    elif isinstance(x, str):
        return write_string_primitive(x)
    
    else:
        raise TypeError(f'Unexpected type: {type(x).__name__}')
    
special_characters = set('!$#`;:,')

def write_string_primitive(x: str) -> str:
    global special_characters
    
    #Check for unbalanced quotes and dunlin syntax characters
    quote                  = []
    has_special_characters = False
    
    for i in x:
        if i in '\'"':
            if not quote:
                quote.append(i)
            elif quote[-1] == i:
                quote.pop(-1)
            else:
                quote.append(i)
        
        elif i in special_characters and not quote:
            has_special_characters = True
            
    
    if quote or has_special_characters:
        return repr(x)
    
    #Check if the readout is a number, boolean or datetime    
    readout = rpr.read_primitive(x)
    
    if not isinstance(readout, str):
        return repr(x)
    
    #Otherwise return x as-is
    return x
