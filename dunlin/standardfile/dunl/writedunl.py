from typing import Union

from .writedictlist import (write_dict, 
                            write_list, 
                            write_primitive, 
                            write_key
                            )
import dunlin.standardfile.dunl.readdunl as rd

###############################################################################
#File Editing
###############################################################################
def write_dunl_file(all_data: dict, filename: str=None, op='write', **kwargs):
    allowed = ['write', 'append', 'merge']
    if op not in allowed:
        raise ValueError(f'op must be one of {allowed}.')
    
    #Generate code
    if filename and op == 'merge':
        with open(filename, 'r') as file:
            other = rd.read_file(filename) 
            
        new_all_data = merge(other, all_data)
        code         = write_dunl_code(new_all_data, **kwargs)
    
    else:
        code = write_dunl_code(all_data, **kwargs)
    
    #Write to file
    if filename:
        if op == 'append':
            with open(filename, 'a') as file:
                file.write(code)
        else:
            with open(filename, 'w') as file:
                file.write(code)
    
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

###############################################################################
#Code Generation
###############################################################################
def write_dunl_code(dct: dict, max_dir: int = 3, indent_type: str = '\t', 
                 multiline_dict: Union[bool, int]=True, _dir: list[str] = ()
                 ) -> str:
    #Use _dir to keep track of the current directory
    
    #Check inputs
    if any([not i.isspace() for i in indent_type]):
        raise ValueError('Invalid indent_type')
    
    #Set up code
    code = ''
    
    #Iterate and update
    for key, value in dct.items():
        if type(value) == dict or hasattr(value, 'to_dunl_dict'):
            if hasattr(value, 'to_dunl_dict'):
                value = value.to_dunl_dict()
                
            #Write the key as directory
            key_      = write_key(key)
            directory = write_directory([*_dir, key_])
            
            #Write the value
            if max_dir:
                m     = max_dir if type(max_dir) == bool else max(0, max_dir - 1)
                chunk = write_dunl_code(value, 
                                     max_dir=m, 
                                     indent_type=indent_type, 
                                     multiline_dict=multiline_dict, 
                                     _dir=[*_dir, key_]
                                     )
            else:
                if not _dir:
                    raise ValueError('Insufficient nesting')
                
                chunk = write_dict(value, 
                                   indent_type=indent_type, 
                                   multiline_dict=multiline_dict
                                   )
            
            #Update the code
            code += directory + '\n' + chunk + '\n\n'
        
        elif not _dir:
            raise ValueError('Insufficient nesting')
        
        elif hasattr(value, 'to_dunl') and _dir:
            #Write the key as directory
            key_      = write_key(key)
            directory = write_directory([*_dir, key_])
            
            #Write the value
            chunk = value.to_dunl()
            
            #Update the code
            code += directory + '\n' + chunk + '\n\n'
         
        else:    
            #Write the key 
            key_   = write_key(key)
            
            #Write the value
            if type(value) in [list, tuple]:
                value_ = write_list(value)
            else:
                value_ = write_primitive(value)
            
            chunk = f'{key_} : {value_}'
            
            #Update the code
            code += chunk + '\n'
                
    code = code.strip()
    return code

def write_directory(directory_lst):
    return ''.join([f';{x}' for x in directory_lst])