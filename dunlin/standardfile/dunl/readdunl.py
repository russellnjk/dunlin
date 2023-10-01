from io      import StringIO
from pathlib import Path
from typing  import Callable

import dunlin.standardfile.dunl.readelement   as rel
import dunlin.standardfile.dunl.parsepath     as pp 
import dunlin.standardfile.dunl.delim         as dm

###############################################################################
#Main Algorithm
###############################################################################
def read_dunl_file(filename: str|Path|StringIO) -> dict:
    
    if isinstance(filename, StringIO):
        lines = filename
        return read_lines(lines)
    
    elif isinstance(filename, (str, Path)):
        path = Path(filename)
        
        if path.is_file() and path.exists():
            with open(path, 'r') as file:
                return read_lines(file)
                
        else:
            lines = path.splitlines()
            return read_lines(lines)
    
    else:
        msg = 'Filename is not a StringIO, string or pathlib.Path.'
        raise ValueError(msg)
    
def read_dunl_code(code: str) -> dict:
    lines = code.splitlines()
    return read_lines(lines)
    
###############################################################################
#Supporting Functions
###############################################################################
def read_lines(lines: list[str]|StringIO) -> dict:
    
    '''Reads an iterable of lines.
    

    Parameters
    ----------
    lines : iterable
        An iterable of strings.
    includes_newline : bool, optional
        True if the lines contain the new line character at the end and False 
        otherwise. The default is False.

    Returns
    -------
    dct : dict
        The parsed data.

    '''
    dct           = {}
    curr_lst      = []
    curr_dct      = None
    interpolators = {}
    chunk         = ''
    
    for line in lines:
        line = remove_comments(line)
        
        if not line:
            continue
        
        elif line[0].isspace():
            chunk += line
            
        else:
            #Read the existing chunk
            curr_dct = read_chunk(dct, curr_lst, curr_dct, interpolators, chunk)
            
            #Start a new chunk
            chunk = line
    
    #Read the existing chunk
    curr_dct = read_chunk(dct, curr_lst, curr_dct, interpolators, chunk)
    
    return dct
   
def read_chunk(dct           : dict, 
               curr_lst      : list, 
               curr_dct      : dict, 
               interpolators : dict, 
               chunk         : str
               ) -> dict:
    '''Update in-place.
    '''
    
    chunk = chunk.strip()
    
    if not chunk:
        pass
    
    elif chunk[0] == '`':
        key, value = read_interpolator(chunk)
        
        interpolators[key] = value
        
    elif chunk[0] == ';':
        #Note that curr_lst is modified in-place
        curr_dct = pp.go_to(dct, chunk, curr_lst)
    
    else:
        if curr_dct is None:
            msg = f'The section for this element has not been set yet\n{chunk}'
            raise SyntaxError(msg)
            
        parsed_element = rel.read_element(chunk, interpolators=interpolators)
        curr_dct.update(parsed_element)
    
    return curr_dct
        
    
def remove_comments(line: str) -> str:
    raw = split_first(line, delimiter='#', expect_present=False)
    
    if raw is None:
        return line
    else:
        cleaned, _ = raw
        return cleaned

def read_interpolator(chunk: str) -> str:
    #Key and value are already stripped
    key, value = split_first(chunk[1:], delimiter='`', expect_present=True)
    
    #Check key and value
    if not key.replace('_', '').isalnum():
        msg = f'Interpolator key can onlz contain alphanumerics or underscores. Received: {key}'
        raise ValueError(msg)
    elif not key[0].isalpha():
        msg = f'Interpolator key must start with alphabet. Received: {key}'
        raise ValueError(msg)
    
    if not value:
        msg = f'Interpolator value for {key} is blank.'
        raise ValueError(msg)
    elif len(value.splitlines()) > 1:
        msg = f'Interpolator value for {key} has more than one line.'
    return key, value

def split_first(string, delimiter=':', expect_present=True) -> str|None:
    quote = []
    for i, char in enumerate(string):
        if char == delimiter and not quote:
            return string[:i].strip(), string[i+1:].strip()
        
        elif char in dm.quotes:
            if not quote:
                quote.append(char)
            elif quote[-1] == char:
                quote.pop()
            else:
                quote.append(char)
    
    if expect_present:
        raise ValueError(f'Missing a "{delimiter}"')
    elif quote:
        raise SyntaxError('Missing a quotation mark to close the string.')
    else:
        return None
    