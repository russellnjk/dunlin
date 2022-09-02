import textwrap as tw
from datetime  import datetime
from numbers   import Number
from pyrfc3339 import generate
from typing    import Union

import dunlin.standardfile.dunl.readprimitive as rpr
from dunlin.utils.typing import Primitive

###############################################################################
#Globals
###############################################################################
_globals = {'indent_type' : '\t'}

def get_indent_type():
    global _globals
    return _globals['indent_type']

def set_indent_type(indent_type: str='\t'):
    global _globals
    _globals['indent_type'] = indent_type
    
###############################################################################
#Writing Containers
###############################################################################
def write_dict(dct, multiline_dict=True, _top_level=True) -> str:
    indent_type = get_indent_type()
    
    code = ''
    
    for key, value in dct.items():
        #Convert the key
        key_ = write_key(key) 
        
        if type(multiline_dict) == bool:
            m = multiline_dict
        else:
            m = max(0, multiline_dict-1)
        #Convert the value
        if type(value) == dict:
            chunk  = write_dict(value, 
                                multiline_dict=m,
                                _top_level=False
                                )
            
            
            if multiline_dict:
                chunk  = tw.indent(chunk, indent_type)
                code  += f'{key_} : [\n' + chunk +  f'\n{indent_type}],\n'
            else:
                code  += f'{key_} : [' + chunk +  '], '
        
        else:
            if type(value) in [list, tuple]:
                chunk = write_list(value)
            else:
                chunk  = write_primitive(value)
                
            chunk  = f'{key_} : {chunk}'
            
            if multiline_dict > 0:
                if _top_level:
                    code += chunk + '\n'
                else:
                    code += chunk + ',\n'
            else:
                code += chunk + ', '
            
    
    #Strip and remove trailing commas
    code = code.strip()
    if code[-1] == ',':
        code = code[:-1]
        
    return code

def write_key(key: Union[Primitive, list]) -> str:
    if type(key) in [list, type]:
        string = write_list(key)
    else:
        string = write_primitive(key)
    
    return string

def write_list(lst) -> str:
    #Open the list
    code = '['
    
    #Iterate and extend
    for x in lst:
        if type(x) == dict:
            chunk = write_dict(x, multiline_dict=False) 
            chunk = '[' + chunk + ']'
        elif type(x) in [list, tuple]:
            chunk = write_list(x)
        else:
            chunk = write_primitive(x)
        code += chunk + ', '
        
    #Remove the trailing comma and close the list
    code = code[:-2] 
    code = code + ']'
    return code

###############################################################################
#Writing Primitives
###############################################################################
def write_primitive(x: Primitive) -> str:
    if isinstance(x, Number):
        return str(x)
    elif type(x) == str:
        if rpr.ismath(x):
            return repr(x)
        elif needs_quotes(x): 
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
    