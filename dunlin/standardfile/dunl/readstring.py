import numpy as np
from datetime import datetime
from numbers  import Number

import dunlin.standardfile.dunl.readprimitive as rpr
import dunlin.standardfile.dunl.delim         as dm

#TODO: find ways to accomodate brackets

###############################################################################
#Key-Value Reader for Substituted dun Strings
###############################################################################
def read_string(string, enforce_dict=True):
    '''
    The front-end algorithm for reading dun strings.

    Parameters
    ----------
    string : str
        The string to be parsed.
    
    Returns
    -------
    result : dict
        A dictionary.
    
    Notes
    -----
    Calls _read_dun. Convert the result to a dict if it was originally a list.
    '''
    try:
        result = _read_string(string, read_flat)
    
    except Exception as e:
        msg = f'Error in parsing string:\n{string}\n'

        raise ExceptionGroup(msg, [e])
    
    if type(result) == list and enforce_dict:
        return dict(enumerate(result))
    else:
        return result
    
def _read_string(string, _flat_reader=lambda x: x):
    '''
    The back-end algorithm for reading dun strings. Should not be directly 
    called outside of testing purposes.

    Parameters
    ----------
    string : str
        The string to be parsed.
    _flat_reader : callable, optional
        The function that parses flattened containers. The default is lambda x: x.

    Returns
    -------
    result : dict, list
        The parsed value. The front-end version will carry out post-processing.
    
    Notes
    -----
    The _flat_reader argument is changed to read_flat in the front-end. Use of 
    the value lambda x: x is for development purposes in which the developer 
    does not need the individual items in the string to be evaluated.
    
    '''
    string  = string.strip()
    i0      = 0
    nested  = []
    curr    = []
    quote   = []
    builtin = False
    
    for i, char in enumerate(string):
        if char == '!' and not quote:
            builtin = not builtin
            
        elif char == dm.item and not quote and not builtin:
            append_chunk(string, i0, i, char, curr)
            
            #Update position
            i0 = i + 1
        
        elif char == dm.pair and not quote and not builtin:
            append_chunk(string, i0, i,   char, curr)
            append_chunk(string, i,  i+1, char, curr)
            
            #Update position
            i0 = i + 1
        
        elif char == dm.open and not quote and not builtin:
            append_chunk(string, i0, i, char, curr)
            
            #Increase depth
            nested.append(curr)
            curr = []
            
            #Update position
            i0 = i + 1
            
        elif char == dm.close and not quote and not builtin:
            if not nested:
                msg = 'Unexpected closing bracket or missing open bracket.'
                msg = f'{msg}\n{excerpt(string, i0, i)}'
                raise SyntaxError(msg)
                
            append_chunk(string, i0, i, char, curr)
            
            #Decrease depth
            parsed = _flat_reader(curr)
            curr   = nested.pop()
            curr.append(parsed)
            
            #Update position
            i0 = i + 1
        
        elif char in dm.quotes:
            if not quote:
                quote.append(char)
            elif quote[-1] == char:
                quote.pop()
            else:
                quote.append(char)
            
        else:
            continue
    
    append_chunk(string, i0, len(string), None, curr)
    
    result = _flat_reader(curr)
    
    if len(nested) or quote or builtin:
        msg = 'Unexpected/missing brackets or delimiters.'
        msg = f'{msg}\n{excerpt(string, i0, len(string))}' 
        raise SyntaxError(msg)
    
    return result

def excerpt(string, i0, i):
    return '...' + string[max(0, i0-10): min(i+10, len(string))] + '...'

def append_chunk(string, i0, i, char, curr):
    '''Updates the current container when a  "[", "]", ":" or ","  is encountered. Checks 
    the sequence of last_char...chunk...char and raises an Exception if values 
    are detected before or after brackets not in accordance with syntax.
    
    :meta private:
    '''
    #Parse the chunk
    chunk = string[i0: i].strip()
    
    #Check for syntax errors from last_char...chunk...char
    #Get the last character that triggered this function
    last_char = string[i0-1] if i0 > 0 else None
    
    #For chunks with content
    if chunk:
        #Characters precede an open bracket e.g. "a ["
        if char == dm.open:
            msg = f'Values before bracket.\n{excerpt(string, i0, i)}'
            raise SyntaxError(msg)
        
        #Characters succeed a close bracket but not a multiplier e.g. "] 2"
        elif last_char == dm.close and chunk[0] != dm.mul:
            msg = f'Values after bracket.\n{excerpt(string, i0, i)}'
            raise SyntaxError(msg)
        
        else:
            curr.append(chunk)
            return chunk
        
    #For consecutive delimiters with only whitespace between
    else:
        #String starts with comma, semicolon or close bracket e.g. ", a"
        if char in [dm.item, dm.pair, dm.close] and i0 == 0:
            msg = f'String starts with a "{char}"\n{excerpt(string, i0, i)}'
            raise SyntaxError(msg)
        
        #Open bracket followed by comma or semicolon  e.g. "[ ,"
        elif last_char == dm.open and char in [dm.item, dm.pair]:
            msg = f'Encountered "{last_char} followed by {char}"\n{excerpt(string, i0, i)}'
            raise SyntaxError(msg)
        
        #Comma/semicolon followed by another comma/semicolon
        elif last_char in [dm.item, dm.pair] and char in [dm.item, dm.pair]:
            msg = f'Encountered illegal consecutive delimiters.\n{excerpt(string, i0, i)}'
            raise SyntaxError(msg)
        
        return
        
###############################################################################
#Reading Flat Containers
###############################################################################
def read_flat(flat):
    '''
    Reads a flat container and evaluates each item if it has not already been 
    evaluated.

    Parameters
    ----------
    flat : list
        A list of chunks to be parsed.

    Returns
    -------
    dict, list
        A dict or list of parsed items.
        
    '''
    if not flat:
        raise ValueError('Encountered an empty container.')
    if ':' in flat:
        return read_dict(flat)
    else:
        return read_list(flat)

def read_dict(flat):
    result = {}
    i0     = 0
    
    while i0 < len(flat):
        key, delim, value, repeat, *_ = *flat[i0: i0+4], '', ''
        
        #Check
        if delim != dm.pair:
            raise SyntaxError('Missing/improper use of semi-colon in a dict.')
        elif repeat[:1] == dm.mul and type(value) != list:
            raise RepeatTypeError()
        
        #Parse
        key   = read_key(key)
        value = read_value(value) if type(value) == str else value
        
        if repeat[:1] == dm.mul:
            #Parse the repeat
            repeat = read_repeat(repeat)
            value  = value*repeat
            
            #Update position
            i0 += 4
        else:
            #Update position
            i0 += 3
        
        #Update result
        result[key] = value
    return result

def read_list(flat):
    result       = [] 
    
    for i, item in enumerate(flat):
        if type(item) == str:
            item = item.strip()
            
            if item[0] == dm.mul:
                repeat = read_repeat(item)
                if not result:
                    raise SyntaxError(f'{repr(dm.mul)} symbol occured before first element.')
                if type(result[-1]) == list:
                    result[-1] = result[-1]*repeat
                else:
                    raise RepeatTypeError()
            else:
                value = read_value(item)
                result.append(value)
        else:
            result.append(item)
    
    return result

def read_repeat(x):
    try:
        return int(x[1:])
    except:
        msg = f'Repeat must be an integer. Received {type(x)}: {x}'
        raise SyntaxError(msg)

class RepeatTypeError(Exception):
    def __init__(self,):
        super().__init__('Repeat can only come after a list.')

class DunMissingError(Exception):
    def __init__(self, msg=''):
        if msg:
            msg = 'Unexpected comma/equal sign or missing key/value.\n' + msg
        else:
            msg = 'Unexpected comma/equal sign or missing key/value.'
        super().__init__(msg)

###############################################################################
#Reading Keys/Values
###############################################################################
def read_key(x):
    '''
    Parses keys. Three cases are possible:
        1. x is a list of primitives.
        2. x represents a dunl builtin function call.
        3. x represents a primitive
    Because Python does not allow lists to used as keys, lists must be converted 
    to tuples.
    '''
    #Case 1: x is a list of primitives
    if type(x) == list:
        key = x
    
    #Case 2 and 3: x represents a primitive or dunl builtin function call
    else:
        key = read_value(x)
            
    if type(key) == list:
        return tuple(key)
    else:
        if type(key) == str:
            if not key.strip():
                raise ValueError('Blank string cannot be used a key.')
        return key

def read_value(x: str) -> Number|str|bool|datetime:
    try:
        return rpr.read_primitive(x)
    except:
        raise DunValueError(x)
   
class DunValueError(Exception):
    def __init__(self, x):
        super().__init__(f'Invalid value: {repr(x)}.')


def split_top_delimiter(string, delimiter=dm.item):
    try:
        string        = string.strip()
    except Exception as e:
        print(string)
        raise e
    i0            = 0
    inside_quotes = False
    chunks        = []
    
    for i, char in enumerate(string):
        if char == delimiter and not inside_quotes:
            
            chunk = string[i0: i].strip()
            
            if chunk:
                if len(chunk) > 1 and chunk[0] == chunk[-1] and chunk[0] in ["'","'"]:
                    chunk = chunk[1:-1]
                chunks.append(chunk)
            else:
                raise ValueError(f'Encountered blank value or extra delimiters: {string}')
                
            i0    = i + 1
        
        elif char in dm.quotes:
            inside_quotes = not inside_quotes

    if i0 < len(string):
        chunk = string[i0: ].strip()
        
        if chunk:
            if len(chunk) > 1 and chunk[0] == chunk[-1] and chunk[0] in dm.quotes:
                chunk = chunk[1:-1]
            chunks.append(chunk)
        else:
            raise ValueError(f'Encountered blank value or extra delimiters: {string}')
            
    return chunks
      