import re
from pathlib import Path

###############################################################################
#Non-Standard Imports
###############################################################################
try:
    from  .base_error  import DunlinBaseError
    from  .custom_eval import safe_eval as eval
except Exception as e:
    if Path.cwd() == Path(__file__).parent:
        from  base_error  import DunlinBaseError
        from  custom_eval import safe_eval  as eval
    else:
        raise e

###############################################################################
#Key-Value Parser for Substituted dun Strings
###############################################################################
def read_dun(string, expect=dict, **kwargs):
    try:
        result = _read_dun(string, **kwargs)
        
        if type(result) != expect:
            raise DunlinStringError.top(expect)
            
    except DunlinStringError as e:
        raise DunlinStringError.merge(e, f'Error in string: {string}')
    except Exception as e:
        raise e
    
    return result

def str2num(x):
    try:
        return int(x)
    except:
        return float(x)

def isnum(x):
    try:
        float(x)
        return True
    except:
        return False
    
def read_value(x):
    try:
        return str2num(x)
    except:
        pass
    
    x_ = x.strip()
    if not x_:
        return None
    #Check if STRING None
    if x_ == 'None':
        return x_
    
    #Check boolean
    if x_ == 'True' or x_ == 'False':
        return eval(x_)
    
    #Check for illegal characters including double underscore
    illegals = [':', ';', '?', '%', '$', 
                '#', '@', '!', '`', '~', '&', 
                '{', '}', '|', '\\', '__',
                ]#The first backslash is an escape character!

    for s in  illegals:
        if s in x_:
            raise DunlinStringError.value(x)
    
    #Check if tuple
    if x_[0] == '(' and x_[-1] == ')':
        return tuple([str2num(i) if isnum(i) else i.strip() for i in  x_[1: len(x_)-1].split(',')])
    
    #Expect a string
    return x_
    
def _read_dun(string, min_depth=None, max_depth=None):
    #Track evaluated
    value   = None
    i0      = 0
    
    #Track depth
    nest  = []
    base  = []
    curr  = base
    nest  = []
    deep  = 0
    
    ignore = 0
    
    for i, c in enumerate(string):
        if c == '[':
            #Collect the datum
            chunk = string[i0:i].strip()
            if chunk:
                curr.append(chunk)
                #Check for stray values
                if ':' not in chunk:
                    raise DunlinStringError.outside('bef')
            
            #Switch depth
            curr.append([])
            nest.append(curr)
            curr = curr[-1]
            
            #Update position
            i0 = i + 1
            
            #Track max depth
            #Check that max_depth is not exceeded
            deep += 1
            if max_depth:
                if deep > max_depth:
                    raise DunlinStringError.depth('max')
            
        elif c == ']':
            #Collect the datum
            chunk = string[i0:i].strip()
            if chunk:
                curr.append(chunk)
                #Check for stray values
                if string[i0-1] == ']':
                    
                    raise DunlinStringError.outside('aft')
            
            #Evaluate
            value = evaluate_level(curr)
            
            #Switch depth
            #Check brackets/nesting
            try:
                curr = nest.pop()
            except:
                raise DunlinStringError.bracket('open')
            
            #Ban blank input
            if value:
                curr[-1] = value
            else:
                raise DunlinStringError.value(value)
            
            #Update position
            i0 = i + 1
            
        elif c == ',':
            if ignore:
                continue
            
            #Collect the datum
            chunk = string[i0:i].strip()
            if chunk:
                curr.append(chunk)
            
            #Update position
            i0 = i + 1
        
        elif c == '(':
            ignore += 1
        elif c == ')':
            ignore -= 1
            
    #Collect the datum
    chunk = string[i0:].strip()
    if chunk:
        curr.append(chunk)
        #Check for stray values
        if string[i0-1] == ']':
            raise DunlinStringError.outside('aft')
    
    #Check brackets/nesting
    if len(nest):
        raise DunlinStringError.bracket('close')
    
    #Evaluate
    value = evaluate_level(curr)
    
    #Ban blank input
    if not value:
        raise DunlinStringError.value(value)
    
    #Check if min_depth is exceeded
    if min_depth:
        if deep < min_depth:
            raise DunlinStringError.depth('min')
    
        
    return value

def evaluate_level(level):

    if not level:
        return
    
    #Account for repaeated lists
    level_ = []
    for i in level:
        if type(i) == str:
            if i[0].strip() == '*':
                level_[-1] == eval(str(level_[-1]) + i)
                continue
            
        level_.append(i)
        
    level = level_
    
    #Parse
    data     = {} if ':' in level[0] else []
    last_key = None if type(data) == dict else '__NA__'

    for i, d in enumerate(level):
        if type(d) == str:
            if type(data) == list:
                if ':' in d:
                    raise DunlinStringError.inconsistent()
                value = read_value(d)
                data.append(value)
            else:
                split = [c.strip() for c in d.split(':', 1)]

                if len(split) == 1:
                    if last_key == None:
                        raise DunlinStringError.inconsistent()
                    else:
                        #Raise
                        raise DunlinStringError.outside('aft')
                else:
                    if last_key == None and split[1]:
                        key, value = split
                        key        = read_value(key)
                        data[key]  = read_value(value)
                    elif last_key == None:
                        last_key = split[0]
                    else:
                        raise DunlinStringError.inconsistent()
        else:
            if type(data) == list:
                data.append(d)
            else:
                if last_key is None:
                    raise DunlinStringError.outside('bef')
                
                key       = read_value(last_key)
                data[key] = d
                last_key  = None
                
    return data

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
        return cls.raise_template('Inconsistent data type.', 1)
    
    @classmethod
    def outside(cls, pos='bef'):
        details = 'Values before bracket.' if pos == 'bef' else 'Values after bracket.'
        return cls.raise_template(details, 2)
    
    @classmethod
    def top(cls, type):
        return cls.raise_template(f'Top level must be {type.__name__}.', 3)
    
    @classmethod
    def bracket(cls, miss='open'):
        if miss == 'open':
            details = 'Detected an unexpected closing bracket or missing opening bracket.'
        else:
            details = 'Detected an unexpected opening bracket or missing closing bracket.'
        return cls.raise_template(details, 4)
    
    @classmethod
    def value(cls, x):
        return cls.raise_template(f'Invalid value: {repr(x)}.', 5)
    
    @classmethod
    def depth(cls, bnd='min'):
        if bnd == 'min':
            details = 'Minimum depth not satisfied.'
        else:
            details = 'Maximum depth exceeded.'
        return cls.raise_template(details, 6)
   
