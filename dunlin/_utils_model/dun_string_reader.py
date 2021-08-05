import re

###############################################################################
#Non-Standard Imports
###############################################################################
from  .base_error  import DunlinBaseError

###############################################################################
#Key-Value readr for Substituted dun Strings
###############################################################################
def read_dun(string, expect=dict, **kwargs):
    try:
        result = _read_dun(string, read_flat, **kwargs)
        
        if type(result) != expect:
            raise DunlinStringError.top(expect)
            
    except DunlinStringError as e:
        raise DunlinStringError.merge(e, f'Error in string: {string}')
    except Exception as e:
        raise e
    
    return result
    
def _read_dun(string, flat_reader=lambda x: x, min_depth=0, max_depth=3):
    string = preprocess_string(string)
    
    i0     = 0
    ignore = False
    nested = []
    curr   = []
    max_d  = 0
    min_d  = min_depth <= 0
    
    for i, char in enumerate(string):
        if char == '(':
            ignore = True
            
        elif char == ')':
            ignore = False
            
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
            
            #Track/check depth
            max_d += 1
            if max_d > max_depth:
                raise DunlinStringError.depth('max')
            if max_d >= min_depth:
                min_d = True
                
        elif char == ']' and not ignore:
            if not nested:
                raise DunlinStringError.bracket('open')
            append_chunk(string, i0, i, char, curr)
            
            #Decrease depth
            parsed = flat_reader(curr)
            curr   = nested.pop()
            curr.append(parsed)
            i0 = i + 1
            
            #Track/check depth
            max_d -= 1
        else:
            continue
    
    append_chunk(string, i0, len(string), None, curr)
    
    result = flat_reader(curr)
    
    if len(nested):
        raise DunlinStringError.bracket('close')
    elif not min_d:
        raise DunlinStringError.depth('min')
    return result

###############################################################################
#Supporting Functions
###############################################################################
def append_chunk(string, i0, i, token, curr):
    chunk      = string[i0: i].strip()
    last_token = string[i0-1] if i0 > 0 else None
    
    if not chunk:
        if token == ',':
            if last_token in [',', '['] or i == 0:
                print(chunk, token, last_token)
                raise DunlinStringError.delimiter()
        
        return 
    
    if token == '[':
        if chunk[-1] != ':':
            raise DunlinStringError.outside('bef')
    
    if token == ',' or token == None:
        if last_token == ']' and chunk[0] != '*':
            raise DunlinStringError.outside('aft')
        
    curr.append(chunk)
    return chunk

def preprocess_string(string):
    new_string = string.strip()
    new_string = new_string[:-1] if new_string[-1] == ',' else new_string
    
    if not new_string:
        raise DunlinStringError.value('')
    
    elif new_string.strip()[-1] == ',':
        raise DunlinStringError.delimiter()
        
    return new_string

def read_flat(flat):
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
                raise DunlinStringError.inconsistent()
                
            s = value_.split(':', 1)
            if len(s) == 1:
                if s[0][0] == '*':
                    last_key = list(key_view)[-1]
                    if type(result[last_key]) == list:
                        repeat           = read_repeat(s[0])
                        result[last_key] = result[last_key]*repeat
                    else:
                        raise DunlinStringError.repeat()
                else:
                    raise DunlinStringError.missing()
            else:
                k, v = read_value(s[0]), read_value(s[1], allow_blank=True)
    
                if v == '':
                    curr_key = k
                else:
                    result[k] = v

        else:
            if curr_key is None:
                raise DunlinStringError.inconsistent()
            else:
                result[curr_key] = value
                curr_key = None
    
    if curr_key is not None:
        raise DunlinStringError.missing()
    return result

def read_list(flat):
    result       = [] 
    
    for i, value in enumerate(flat):
        if type(value) == str:
            value_ = value.strip()
            if not value_:
                raise Exception('Blank string.')
            elif ':' in value:
                raise DunlinStringError.inconsistent()
            elif value_[0] == '*':
                repeat = read_repeat(value_)
                if type(result[-1]) == list:
                    result[-1] = result[-1]*repeat
                else:
                    raise DunlinStringError.repeat()
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
        raise DunlinStringError.invalid_repeat()

def read_value(x, allow_blank=False):
    if type(x) != str:
        raise DunlinStringError.value(x)
    
    #Try num
    try:
        return int(x)
    except:
        try:
            return float(x)
        except:
            pass
        
    x_ = x.strip()
    if not x_ and not allow_blank:
        raise DunlinStringError.blank_key()
        
    #Try boolean
    if x_ == 'True' or x_ == 'False':
        return eval(x_)
    
    #Expect a string
    #Check for illegal characters including double underscore
    illegals = [':', ';', '?', '%', '$', 
                '#', '@', '!', '`', '~', '&', 
                '{', '}', '|', '\\', '__',
                ]#The first backslash is an escape character!

    for s in  illegals:
        if s in x_:
            raise DunlinStringError.value(x_)
    
    return x_

###############################################################################
#Key-Value Parser for CLEANED PY Strings
###############################################################################
def format_indent(string):
    try:
        name, code = string.split(':', 1)
    except:
        raise DunlinStringError.py(string)
        
    name = name.strip()
    
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

###############################################################################
#Supporting Functions
###############################################################################
def validate_key(key):
    if '__' in key:
        raise NameError(f'Found double underscores in key: {key}')
    elif '=' in key:
        raise NameError(f'Found equals sign in key: {key}')

    return read_value(key)

###############################################################################
#Dunlin Exceptions
###############################################################################
class DunlinStringError(SyntaxError, DunlinBaseError):
    @classmethod
    def inconsistent(cls):
        return cls.raise_template('Inconsistent data type.', 0)
    
    @classmethod
    def outside(cls, pos='bef'):
        details = 'Values before bracket.' if pos == 'bef' else 'Values after bracket.'
        return cls.raise_template(details, 1)
    
    @classmethod
    def delimiter(cls):
        return cls.raise_template('Unexpected delimiter', 2)
    
    @classmethod
    def repeat(cls):
        return cls.raise_template('Repeat can only come after a list', 3)
    
    @classmethod
    def invalid_repeat(cls):
        return cls.raise_template('Repeat must be an integer', 4)
    
    @classmethod
    def missing(cls):
        return cls.raise_template('Unexpected comma/colon or missing key/value', 5)
        
    @classmethod
    def bracket(cls, miss='open'):
        if miss == 'open':
            details = 'Detected an unexpected closing bracket or missing opening bracket.'
        else:
            details = 'Detected an unexpected opening bracket or missing closing bracket.'
        return cls.raise_template(details, 6)

    @classmethod
    def value(cls, x):
        return cls.raise_template(f'Invalid value: {repr(x)}.', 7)
    
    @classmethod
    def blank_key(cls, x):
        return cls.raise_template('Keys cannot be blank', 8)
    
    @classmethod
    def depth(cls, bnd='min'):
        if bnd == 'min':
            details = 'Minimum depth not satisfied.'
        else:
            details = 'Maximum depth exceeded.'
        return cls.raise_template(details, 9)
   
    @classmethod
    def top(cls, type):
        return cls.raise_template(f'Top level must be {type.__name__}.', 10)
    

