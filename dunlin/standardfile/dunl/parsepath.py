from datetime import datetime

import dunlin.standardfile.dunl.readprimitive as rpr
import dunlin.standardfile.dunl.readstring    as rst
import dunlin.standardfile.dunl.delim         as dm

def go_to(dct, path, curr_lst, assign=None):
    '''
    Recurses through a dictionary to reach a sub-dictionary, analagous to 
    traversing directories in paths in a system of folders.  

    Parameters
    ----------
    dct : dict
        The dictionary that will be accessed.
    path : str or list
        If `path` is a string, it consists of keys separated by delimiters. The 
        first character of the path must be the delimiter. For example `.a.b`. 
        Alternatively, `path` can be a list of keys. 
    curr_lst : list of keys
        A list of keys corresponding to the current directory. Allows the function 
        to parse relative paths such as `..b` which means b is a subdirectory 
        of the current directory.
    assign : TYPE, optional
        The value to assign to the directory. If not None, the directory is 
        replaced with the value of this argument. The default is None.

    Notes
    -----
    Dunlin's language does not make use of the assign argument. May be removed 
    in the future.
    
    Important: `curr_lst` is modified in place. 
    
    Returns
    -------
    dst : dict
        Returns the subdirectory. It will be created if it does not already exist. 
        If the `assign` argument is used, it will be the value of `assign` instead 
        of a dictionary.
    curr_lst : list of str
        A list of keys corresponding to the current directory.

    '''
    #Preprocess
    path     = split_path(path) if type(path) == str else path
    path_lst = replace_relative_path(path, curr_lst)
    
    #Recurse and update curr_lst
    dst = recurse(dct, path_lst, assign)
    curr_lst.clear()
    curr_lst.extend(path_lst)
    
    return dst, curr_lst

def split_path(string):
    '''
    Splits a path-like string into keys. Each key must be a valid primitive in 
    dunlin's language. The delimiter is ignored if it occurs inside a pair of 
    quotes.
    
    Notes
    -----
    The delimiter is expected to occur at the start of the string.
    '''
    string = string.strip()
    if not string:
        raise ValueError('Blank path.')
    elif string[0] != dm.directory:
        raise ValueError('Directory must begin with delimiter.')
        
    string = string[1:]
    i0     = 0
    quote  = []
    chunks = []
    
    for i, char in enumerate(string):
        if char == dm.directory and not quote:
            chunk = string[i0:i]
            chunk = read_key(chunk)
                
            chunks.append(chunk)
            
            i0 = i + 1
        
        elif char in dm.quotes:
            if not quote:
                quote.append(char)
            elif quote[-1] == char:
                quote.pop()
            else:
                quote.append(char)
    
    chunk = string[i0:]
    chunk = read_key(chunk)
    chunks.append(chunk)
    
    return chunks

def read_key(string):
    '''
    Parses the string into a key.
    '''
    string = string.strip()
    
    if not string:
        return ''

    temp   = rst.read_string(string, enforce_dict=False)
    
    #Check that temp is a list
    if type(temp) == dict:
        msg = f'Error in reading {string}.'
        msg = f'{msg} Dict cannot be used in a directory.'
        raise TypeError(msg)
    elif len(temp) != 1:
        msg = f'Error in reading {string}.'
        msg = f'{msg} Missing one or more brackets.'
        raise ValueError(msg)
        
    #Extract the raw key from the list
    key = temp[0]
    
    if type(key) == list:
        key = tuple(key)
    elif type(key) == dict:
        msg = f'Error in reading {string}.'
        msg = f'{msg} Dict cannot be used in a directory.'
        raise TypeError(msg)
    elif type(key) == str:
        if not key.strip():
            raise ValueError('Blank string cannot be used a key.')
        
    return key
    
def replace_relative_path(path, curr_lst):
    '''Converts path from a string into a list. Replaces the blanks in the path 
    with values from curr_lst.
    '''
    path_       = []
    allow_blank = True
    
    for i, p in enumerate(path):
        if type(p) == str and not p:
            if allow_blank:
                if i >= len(curr_lst):
                    raise ValueError(f'Directory missing {path}')
                    
                s = curr_lst[i]
                path_.append(s)
                
        else:
            path_.append(p)
            allow_blank = False

    return path_

def recurse(dct, path_lst, assign=None):
    '''The recursive function for accessing the subdirectory. Creates new 
    directories (dicts) if they do not already exist.
    '''
    #Get next level
    next_level = path_lst[0]
    dct_       = dct.setdefault(next_level, type(dct)())
    
    if len(path_lst) == 1:
        if assign is not None:
            dct[next_level] = assign
            return assign
        else:
            return dct_
        
    else:
        path_lst_ = path_lst[1:]

        return recurse(dct_, path_lst_, assign)
