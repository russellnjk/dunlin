import numpy as np
from numbers     import Number
from string      import ascii_letters, Formatter

###############################################################################
#Top-level Function
###############################################################################
def read_shorthand(interpolated: str) -> list[str]:
    interpolated_ = read_horizontal(interpolated)
    strings       = read_vertical(interpolated_)
    
    return strings

###############################################################################
#Horizontal Shorthands
###############################################################################
def read_horizontal(interpolated) -> str:
    global quotes
    
    i0            = 0
    quote         = []
    in_horizontal = False
    result        = ''
    
    for i, char in enumerate(interpolated):
        #Keep track of quotes
        if char in quotes:
            if not quote:
                quote.append(char)
            elif quote[-1] == char:
                quote.pop()
            else:
                quote.append(char)
        
        
        elif char == '!' and not quote:
            chunk = interpolated[i0:i]
            
            if in_horizontal:
                chunk = _read_horizontal(chunk)
                
            result += chunk
            
            i0 = i + 1
            in_horizontal = not in_horizontal
            
    chunk   = interpolated[i0:]
    result += chunk
    
    return result

def try_number(x: str, integer: bool=False) -> float|int|str:
    '''Attempts to convert a string into a number. If it fails, this function 
    returns a stripped version of the string.
    '''
    try:
        f = float(x)
        
        if integer:
            return int(f)
        else:
            return f
    except:
        return x.strip()
    
def _read_horizontal(chunk: str) -> str:
    args = [i.strip() for i in chunk.split(',')]
    
    match args:
        case 'range', start, stop, step, *extras:
            new_chunk = make_range(start, stop, step, *extras)
        
        case 'linspace', start, stop, step, *extras:
            new_chunk = make_linspace(start, stop, step, *extras)
            
        case 'comma', 'zip', template, *inputs:
            new_chunk = make_zipped(template, *inputs, delimiter=', ')
        
        case 'plus', 'zip', template, *inputs:
            new_chunk = make_zipped(template, *inputs, delimiter=' + ')
        
        case 'multiply', 'zip', template, *inputs:
            new_chunk = make_zipped(template, *inputs, delimiter=' * ')
            
        case 'comma', template, *inputs:
            new_chunk = make_joined(template, *inputs, delimiter=', ')
        
        case 'plus', template, *inputs:
            new_chunk = make_joined(template, *inputs, delimiter=' + ')
        
        case 'multiply', template, *inputs:
            new_chunk = make_joined(template, *inputs, delimiter=' * ')
            
        case _:
            msg = f'Could not parse horizontal shorthand {chunk}'
            raise NotImplementedError(msg)
            
    return new_chunk
        
def make_range(start: str, stop: str, step: str, *extras) -> str:
    start_  = try_number(start)
    stop_   = try_number(stop)
    step_   = try_number(step)
    extras_ = [try_number(i) for i in extras]
    
    all_args = [start_, stop_, *extras_]
    
    if all([isinstance(i, Number) for i in all_args]):
        array = np.arange(start_, stop_, step_)
        array = np.concatenate((array, extras_))
        array = np.unique(array)
        
        to_join   = ['{:.6f}'.format(i) for i in array]
        new_chunk = ', '.join(to_join)
        
        return new_chunk
        
    elif all([isinstance(i, str) for i in all_args]):
        
        to_join = []
        add     = False
        step_   = int(step_)
        
        if start_ not in ascii_letters:
            s    = f'{[start, stop, step, *extras]}'
            msg  = f'Error parsing range shorthand with arguments {s}. '
            msg += f'{start_} not in ascii_letters.'
            raise ValueError(msg)
            
        elif stop_ not in ascii_letters:
            s    = f'{[start, stop, step, *extras]}'
            msg  = f'Error parsing range shorthand with arguments {s}. '
            msg += f'{stop_} not in ascii_letters.'
            raise ValueError(msg)
            
        
        for i in ascii_letters[::step_]:
            
            if i == start_:
                add = True
            elif i == stop_:
                add = False
                
            if add:
                to_join.append(i)
        
        to_join   += sorted(extras_)
        new_chunk  = ', '.join(to_join)
        
        return new_chunk
        
    else:
        s    = f'{[start, stop, step, *extras]}'
        msg  = f'Error parsing range shorthand with arguments {s}. '
        msg += 'Arguments must be all numbers or all strings. '
        raise ValueError(msg)

def make_linspace(start: str, stop: str, step: str, *extras) -> str:
    start_  = try_number(start)
    stop_   = try_number(stop)
    step_   = try_number(step, integer=True)
    extras_ = [try_number(i) for i in extras]
    
    all_args = [start_, stop_, *extras_]
    
    if all([isinstance(i, Number) for i in all_args]):
        array = np.linspace(start_, stop_, step_)
        array = np.concatenate((array, extras_))
        array = np.unique(array)
        
        to_join   = ['{:.6f}'.format(i) for i in array]
        new_chunk = ', '.join(to_join)
        
        return new_chunk
    
    else:
        s    = f'{[start, stop, step, *extras]}'
        msg  = f'Error parsing linspace shorthand with arguments {s}. '
        msg += 'Arguments must be all numbers. '
        raise ValueError(msg)

def make_joined(template: str, *inputs, delimiter=', ') -> str:
    template = template.strip()
    n_fields = len([i for i in Formatter().parse(template) if i[1] is not None])
    to_join  = []
    
    if n_fields == 0:
        a   = f'{[template, *inputs]}'
        msg = f'Error parsing joined shorthand with args {a}. The template has zero fields.'
        raise ValueError(msg)
    
    i = 0
    while i < len(inputs):
        to_sub = []
        for s in inputs[i: i+n_fields]:
            s_ = s.strip() 
            
            if not s_:
                a   = f'{[template, *inputs]}'
                msg = f'Encountered a blank value in joined shorthand with args {a}.'
                raise ValueError(msg)
            else:
                to_sub.append(s_)
            
        new = template.format(*to_sub)
        to_join.append(new)
        i += n_fields
    
    new_chunk = delimiter.join(to_join)
    
    return new_chunk

def make_zipped(template: str, *inputs, delimiter=', ') -> str:
    template = template.strip()
    n_fields = len([i for i in Formatter().parse(template) if i[1] is not None])
    to_join  = []
    
    if n_fields == 0:
        a   = f'{[template, *inputs]}'
        msg = f'Error parsing zipped shorthand with args {a}. The template has zero fields.'
        raise ValueError(msg)
    
    stride  = len(inputs)//n_fields
    inputs_ = [inputs[i*stride:(i+1)*stride] for i in range(n_fields)]
    
    for group in zip(*inputs_):
        to_sub = []
        for s in group:
            s_ = s.strip() 
            
            if not s_:
                a   = f'{[template, *inputs]}'
                msg = f'Encountered a blank value in joined shorthand with args {a}.'
                raise ValueError(msg)
            else:
                to_sub.append(s_)
            
        new = template.format(*to_sub)
        to_join.append(new)
        
    new_chunk = delimiter.join(to_join)
    
    return new_chunk

###############################################################################
#Vertical Shorthands
###############################################################################
def read_vertical(interpolated: str) -> list[str]:
    template, shorthands = split_interpolated(interpolated)
    
    if not shorthands:
        return [template]
    
    strings = []
    keys    = list(shorthands)
    
    try:
        values  = list(zip(*shorthands.values(), strict=True))
    except Exception as e:
        msg = f'Error parsing vertical shorthand for: {interpolated}'
        
        raise ExceptionGroup(msg, [e])
    
    for row in values:
        to_sub = dict(zip(keys, row))
        string = template.format(**to_sub)
        
        #Make sure there are no fields after substitution
        n_fields = [i for i in Formatter().parse(string) if i[1]]
        
        if n_fields:
            msg  = f'Error parsing vertical shorthand for: {interpolated}'
            msg += 'Detected unused fields even after substitution.'
            raise ValueError(msg)
        
        strings.append(string)
    
    return strings
    
quotes = '\'"'

def split_interpolated(interpolated: str) -> tuple[str, str]:
    global quotes
    
    def get_chunk(interpolated, i0, i):
        raw = interpolated[i0: i].split(',')
        
        chunk = []
        for i in raw:
            i_ = i.strip()
            
            if not i:
                msg = f'Encountered a blank value in chunk {raw}.'
                raise ValueError(msg)
            else:
                chunk.append(i_)
        
        if chunk:
            return chunk
        else:
            msg  = f'Error parsing element: {interpolated}\n'
            msg += f'Detected a shorthand with {key} but missing values.'
            raise ValueError(msg)
        
    def check_key(key):
        if not key:
            msg  = f'Error parsing element: {interpolated}\n'
            msg += 'Detected a blank key.'
            raise ValueError(msg)
    
    template   = None
    i0         = 0
    quote      = []
    key        = None
    shorthands = {}

    for i, char in enumerate(interpolated):
        #Keep track of quotes
        if char in quotes:
            if not quote:
                quote.append(char)
            elif quote[-1] == char:
                quote.pop()
            else:
                quote.append(char)
                
        #The shorthand_type is being parsed
        elif char == '$' and not quote:
            #Check if template has been identified
            #If no, update template
            if template is None:
                template = interpolated[:i].strip()

            if key is None:
                i0              = i + 1
                
            else:
                chunk           = get_chunk(interpolated, i0, i,)
                shorthands[key] = chunk
                
                i0             = i + 1
                key            = None
        
        elif char == ':' and not quote and template is not None and key is None:
            key = interpolated[i0:i].strip()
            check_key(key)
            i0  = i + 1
            
        else:
            pass
    
    if template is None:
        template = interpolated
        
    elif key is None:
        msg  = f'Error parsing element: {interpolated}\n'
        msg += 'Incomplete shorthand.'
        raise ValueError(msg)
        
    else:
        chunk           = get_chunk(interpolated, i0, len(interpolated))
        shorthands[key] = chunk
     
    return template, shorthands
