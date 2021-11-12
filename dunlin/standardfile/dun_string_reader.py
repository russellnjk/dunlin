import numpy as np
import re

###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin._utils_plot as upp

###############################################################################
#Key-Value Reader for Substituted dun Strings
###############################################################################
def read_dun(string):
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
    result = _read_dun(string, read_flat)
        
    if type(result) == list:
        return dict(enumerate(result))
    else:
        return result
    
def _read_dun(string, _flat_reader=lambda x: x):
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
    string = preprocess_string(string)
    i0     = 0
    ignore = 0
    nested = []
    curr   = []
    
    for i, char in enumerate(string):
        if char == '(':
            ignore += 1
            
        elif char == ')':
            ignore -= 1
            
        elif char == ',' and not ignore:
            append_chunk(string, i0, i, char, curr)
            
            #Update position
            i0 = i + 1
        
        elif char == '[' and not ignore:
            append_chunk(string, i0, i, char, curr)
            
            #Increase depth
            nested.append(curr)
            curr = []
            
            #Update position
            i0 = i + 1
            
        elif char == ']' and not ignore:
            if not nested:
                raise DunBracketError('open')
            append_chunk(string, i0, i, char, curr)
            
            #Decrease depth
            parsed = _flat_reader(curr)
            curr   = nested.pop()
            curr.append(parsed)
            i0 = i + 1

        else:
            continue
    
    append_chunk(string, i0, len(string), None, curr)
    
    result = _flat_reader(curr)
    
    if len(nested):
        raise DunBracketError('close')

    return result

def preprocess_string(string):
    '''Strips the string and removes trailing commas that would complicate the 
    parsing process.
    '''
    new_string = string.strip()
    new_string = new_string[:-1] if new_string[-1] == ',' else new_string
    
    if not new_string:
        raise DunValueError(new_string)
    
    elif new_string.strip()[-1] == ',':
        raise DunDelimiterError()
        
    return new_string

def append_chunk(string, i0, i, token, curr):
    '''Updates the current container when a "]" or "," is encountered.
    :meta private:
    '''
    chunk      = string[i0: i].strip()
    last_token = string[i0-1] if i0 > 0 else None
    
    if not chunk:
        if token == ',':
            if last_token in [',', '['] or i == 0:
                raise DunDelimiterError()
        
        return 
    
    if token == '[':
        if chunk[-1] != ':':
            raise DunOutsideError('bef')
    
    if token == ',' or token == None:
        if last_token == ']' and chunk[0] != '*':
            raise DunOutsideError('aft')
        
    curr.append(chunk)
    return chunk

class DunOutsideError(Exception):
    def __init__(self, pos='bef'):
        details = 'Values before bracket.' if pos == 'bef' else 'Values after bracket.'
        super().__init__(details)

class DunDelimiterError(Exception):
    def __init__(self):
        super().__init__('Unexpected delimiter.')

class DunBracketError(Exception):
    def __init__(self, miss='open'):
        if miss == 'open':
            details = 'Detected an unexpected closing bracket or missing opening bracket.'
        else:
            details = 'Detected an unexpected opening bracket or missing closing bracket.'
        
        super().__init__(details)
        
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
    if ':' in flat[0]:
        return read_dict(flat)
        
    else:
        return read_list(flat)

def read_dict(flat):
    result   = {}
    curr_key = None
    key_view = result.keys()
    
    for i, value in enumerate(flat):
        if type(value) == str:
            value_ = value.strip()
            if not value_:
                raise Exception('Blank string.')
            
            if curr_key is not None:
                raise DunInconsistentError()
                
            s = value_.split(':', 1)
            if len(s) == 1:
                if s[0][0] == '*':
                    last_key = list(key_view)[-1]
                    if type(result[last_key]) == list:
                        repeat           = read_repeat(s[0])
                        result[last_key] = result[last_key]*repeat
                    else:
                        raise DunRepeatTypeError()
                else:
                    raise DunMissingError()
            else:
                k, v = read_key(s[0]), read_value(s[1])
    
                if v == '':
                    curr_key = k
                else:
                    result[k] = v

        else:
            if curr_key is None:
                raise DunInconsistentError()
            else:
                result[curr_key] = value
                curr_key = None
    
    if curr_key is not None:
        raise DunMissingError()
    return result

def read_list(flat):
    result       = [] 
    
    for i, value in enumerate(flat):
        if type(value) == str:
            value_ = value.strip()
            if not value_:
                raise Exception('Blank string.')
            elif ':' in value:
                raise DunInconsistentError()
            elif value_[0] == '*':
                repeat = read_repeat(value_)
                if type(result[-1]) == list:
                    result[-1] = result[-1]*repeat
                else:
                    raise DunRepeatTypeError()
            else:
                value_ = read_value(value_)
                result.append(value_)
        else:
            result.append(value)
    
    return result

def read_repeat(x):
    try:
        return int(x[1:])
    except:
        raise DunRepeatInvalidError()

class DunInconsistentError(Exception):
    def __init__(self):
        super().__init__('Inconsistent data type.')

class DunRepeatTypeError(Exception):
    def __init__(self,):
        super().__init__('Repeat can only come after a list.')

class DunRepeatInvalidError(Exception):
    def __init__(self):
        super().__init__('Repeat must be an integer.')

class DunMissingError(Exception):
    def __init__(self,):
        super().__init__('Unexpected comma/colon or missing key/value')

###############################################################################
#Reading Keys/Values
###############################################################################
def read_key(key):
    '''Parses keys while enforcing naming conventions. E.g. no lists.
    '''
    if '=' in key:
        raise DunKeyError(key)
    try:
        return read_value(key)
    except DunValueError:
        raise DunKeyError(key)

def read_value(x):
    '''
    Evaluates a single item from a dun string. 

    Parameters
    ----------
    x : str
        A string where:
            1. x is a number.
            2. x is a keyword function call.
            3. x is a string
            4. x is math expression
            5. x is a tuple of the above
            

    Returns
    -------
    x
        The evaluated value of x.

    '''
    x = x.strip()
    if not x:
        return ''
    elif x[0] == '(' and ',' in x:
        return read_tuple(x)
    else:
        return read_single(x)
    
def read_tuple(x):
    '''Evaluates a single item from a dun string where that item is a tuple. 
    Called from read_value.
    '''
    return tuple([read_single(x_) for x_ in x[1:len(x)-1].split(',')])
    
def read_single(x):
    '''
    Reads a single value "x" which corresponds to one of the 4 cases:
        1. x is a number.
        2. x is a keyword function call.
        3. x is a string
        4. x is math expression
    
    Notes
    -----
    Called from read_value.
    The value must be stripped before it is passed into this function.
    
    '''
    
    #Ensure input is a string
    if type(x) != str:
        raise DunValueError(x)
    
    #Case 1: x is a number
    try:
        return int(x)
    except:
        try:
            return float(x)
        except:
            pass
    
    #Check for illegal symbols
    illegals = [':', ';', '?', '%', '$', 
                '#', '@', '!', '`', '~', '&', 
                '{', '}', '|',  
                '\\', '__', '"', "'"
                ]
    for s in  illegals:
        if s in x:
            raise DunValueError(x)
    
    #Case 2: x is "keyword" function call
    try:
        return eval_keyword_func(x)
    except InvalidFunctionError:
        pass
    except Exception as e:
        raise e
    
    #Case 3: x is a math expression
    try:
        r = eval(x, {})
        
        if isnum(r):
            return r
        else:
            raise DunValueError(r)
    except:
        pass
    
    #Case 4: x is a string
    s = x.strip()
    if '(' in s or ')' in s:
        raise DunValueError(s)
    else:
        return s
    
def isnum(x):
    try:
        float(x)
        return True
    except:
        return False

class DunValueError(Exception):
    def __init__(self, x):
        super().__init__(f'Invalid value: {repr(x)}.')

class DunKeyError(Exception):
    def __init__(self, x):
        super().__init__(f'Invalid key: {repr(x)}.')
        
class DunBlankKeyError(Exception):
    def __init__(self):
        super().__init__('Detected a blank key.')
        
class DunExpectedTypeError(Exception):
    def __init__(self, expect):
        super().__init__(f'This element is supposed to be a {expect.__name__}')

###############################################################################
#Keyword Functions
###############################################################################
def eval_keyword_func(x):
    '''Evaluates the string as function call.
    '''
    try:
        if x[:8] == 'linspace':
            return eval(x, {}, {'linspace': linspace})
                
        elif x[:13] == 'light_palette':
            args = x[14: len(x)-1]
            args = _read_dun(args, read_flat)
            
            if type(args) == dict:
                return upp.make_light_scenarios(**args)
            else:
                return upp.make_light_scenarios(*args)
        
        elif x[:12] == 'dark_palette':
            args = x[13: len(x)-1]
            args = _read_dun(args, read_flat)
            
            if type(args) == dict:
                return upp.make_dark_scenarios(**args)
            else:
                return upp.make_dark_scenarios(*args)
        
        elif x[:13] == 'color_palette':
            args = x[14: len(x)-1]
            args = _read_dun(args, read_flat)
            
            if type(args) == dict:
                return upp.make_color_scenarios(**args)
            else:
                return upp.make_color_scenarios(*args)
            
        else:
            raise InvalidFunctionError()
    except InvalidFunctionError as e:
        raise e
    except:
        raise KeywordFunctionError(x)

def linspace(start, stop, *args):
    '''
    Creates a list of time points. Meant to be used as a shortcut for creating 
    tspan.

    Parameters
    ----------
    start : float-like
        The starting point.
    stop : float-like
        The last point.
    *args : list-like
        List or tuple of time points.

    Returns
    -------
    list
        A list of sorted time points.
    
    Notes
    -----
    Calls numpy.linspace to generate values and sorts them with nump.unique.
    
    '''
    n      = args[0] if args else 11
    points = args[1:]  
    lst    = list(np.linspace(start, stop, n)) + list(points)
    return list(np.unique(lst))

class InvalidFunctionError(Exception):
    pass

class KeywordFunctionError(Exception):
    def __init__(self, x):
        msg = f'Could not evaluate the following string containing a keyword function. Your syntax may be wrong. {x}'
        super().__init__(msg)
    
###############################################################################
#Key-Value Parser for CLEANED PY Strings
###############################################################################
def format_indent(string):
    '''
    Attempts to format the first line of the string based on the indentation of 
    the second line. For safety, the first line should not be used anyway.

    Parameters
    ----------
    string : string
        A py string.

    Returns
    -------
    name
        The name for the dun element which will be used downstream.
    formatted
        The formatted string.

    '''
    try:
        name, code = string.split(':', 1)
    except:
        raise DunPyError(string)
        
    name  = name.strip()
    lines = code.split('\n')
    
    if len(lines) == 1:
        return name, '\t' + lines[0].strip()
    
    elif not lines[0].strip():
        return name, '\n'.join(lines[1:])

    for i, line in enumerate(lines[1:], 1):
        if line.strip():
            break
    
    first_indent = re.search('\s*', line)[0]
    line0        = first_indent + lines[0].strip()
    formatted    = '\n'.join([line0, *lines[1:]])

    return name, formatted

class DunPyError(Exception):
    def __init__(self, string):
        super().__init__(f'Could not identify Python code. May be missing a colon.\n {string}')

