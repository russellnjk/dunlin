from typing import Union

from .writecode import write_dunl_code
import dunlin.standardfile.dunl.readdunl as rd

###############################################################################
#File Editing
###############################################################################
def write_dunl_file(all_data: dict, filename: str=None, op='write', **kwargs):
    allowed = ['write', 'append', 'merge']
    if op not in allowed:
        raise ValueError(f'op must be one of {allowed}.')
    
    #Generate code
    if filename:
        if op == 'merge':
            with open(filename, 'r') as file:
                other = rd.read_file(filename) 
                
            new_all_data = merge(other, all_data)
            code         = write_dunl_code(new_all_data, **kwargs)
        
        elif op == 'append':
            code = write_dunl_code(all_data, **kwargs)
            with open(filename, 'a') as file:
                file.write(code)
        else:
            code = write_dunl_code(all_data, **kwargs)
            with open(filename, 'w') as file:
                file.write(code)
    else:
        code = write_dunl_code(all_data, **kwargs)
    
    return code

###############################################################################
#Dict-Merging
###############################################################################
def merge(old: dict, new: dict) -> dict:
    '''Deep-merges two dictionaries
    '''
    result = {}
    seen   = set()
    for key in old:
        seen.add(key)
        if key in new:
            if type(old[key]) == dict and type(new[key]) == dict:
                result[key] = merge(old[key], new[key])
            else:
                result[key] = new[key]
        else:
            result[key] = old[key]
    
    for key in new:
        if key not in seen:
            result[key] = new[key]
            
    return result
