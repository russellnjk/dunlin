from pathlib import Path

import dunlin.standardfile.dunl.readelement   as rel
import dunlin.standardfile.dunl.parsepath     as pp 
import dunlin.standardfile.dunl.delim         as dm

###############################################################################
#Main Algorithm
###############################################################################
def read_dunl_file(*filenames):
    lines = []
    for filename in filenames:
        if type(filename) in [str, Path]:
            with open(filename, 'r') as file:
                lines += file.readlines()
        else:
            #Assumes a text io
            lines += filename.readlines()
    
    dct = read_lines(lines, includes_newline=True, _element=rel.read_element)
    return dct
    
def read_dunl_code(code):
    lines = code.split('\n')
    return read_lines(lines, _element=rel.read_element)
    
###############################################################################
#Supporting Functions
###############################################################################
def read_lines(lines, includes_newline=False, _element=rel.read_element):
    '''Reads an iterable of lines.
    

    Parameters
    ----------
    lines : iterable
        An iterable of strings.
    includes_newline : bool, optional
        True if the lines contain the new line character at the end and False 
        otherwise. The default is False.
    _element : callable
        The function for parsing elements.
    
    Returns
    -------
    dct : dict
        The parsed data.

    '''
    dct           = {}
    curr_lst      = []
    curr_dct      = None
    interpolators = {}
    chunk_lst     = []
    join          = '' if includes_newline else '\n'
    
    for line in lines:
        split = split_first(line, delimiter='#', expect_present=False)
        if split:
            line = split[0]
        
        if not line:
            chunk_lst.append(line)
        elif line[0].isspace():
            chunk_lst.append(line)
        else:
            if chunk_lst:
                chunk    = join.join(chunk_lst)
                curr_dct = read_chunk(dct, curr_lst, curr_dct, interpolators, chunk, rel.read_element)
            
            chunk_lst.clear()
            chunk_lst.append(line)
    
    if chunk_lst:
        chunk    = join.join(chunk_lst)
        curr_dct = read_chunk(dct, curr_lst, curr_dct, interpolators, chunk, rel.read_element)
    
    return dct
    
def read_chunk(dct, curr_lst, curr_dct, interpolators, chunk, _element=rel.read_element):
    chunk  = chunk
    chunk_ = chunk.strip()
    
    if not chunk_:
        pass
    
    
    elif chunk[0] == '`':
        split = split_first(chunk, expect_present=False)
        
        if split is None:
            # raise SyntaxError(f'Invalid interpolators. {chunk}')
            if curr_dct is None:
                msg = f'The section for this element has not been set yet\n{chunk}'
                raise SyntaxError(msg)
                
            parsed_element = _element(chunk, interpolators=interpolators)
            curr_dct.update(parsed_element)
            
            
        elif len(split) != 2:
            raise SyntaxError(f'Invalid interpolators. {chunk}')
        else:
            key, value = split
            key        = key[1:-1].strip()
            value      = value.strip()
            
            interpolators[key] = value
    
    elif chunk[0] == ';':
        split = split_first(chunk, expect_present=False)
        
        if split is not None:
            msg = 'Could not determine if this chunk is a directory or an element.'
            msg = f'{msg} There appears to be both colons and semicolons used.\n{chunk}'
            raise SyntaxError(msg)

        curr_dct, curr_lst = pp.go_to(dct, chunk, curr_lst)
        
    else:
        if curr_dct is None:
            msg = f'The section for this element has not been set yet\n{chunk}'
            raise SyntaxError(msg)
            
        parsed_element = _element(chunk, interpolators=interpolators)
        curr_dct.update(parsed_element)
    
    return curr_dct

def split_first(string, delimiter=dm.pair, expect_present=True):
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
    