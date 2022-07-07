import textwrap as tw

import dunlin.utils                      as ut
import dunlin.standardfile.dunl.readdunl as rd

###############################################################################
#Front-end Functions
###############################################################################
def write_all_data(all_data, filename=None, op='write', indent_type='\t', max_dir=2):
    #Check inputs
    if any([not i.isspace() for i in indent_type]):
        raise ValueError('Invalid indent_type')
    
    allowed = ['write', 'append', 'merge']
    if op not in allowed:
        raise ValueError(f'op must be one of {allowed}.')
    
    #Generate code
    if filename and op == 'merge':
        with open(filename, 'r') as file:
            other = rd.read_file(filename) 
            
        new_all_data = merge(other, all_data)
        code         = write_dict_directory(new_all_data, indent_type, max_dir)
    
    else:
        code = write_dict_directory(all_data, indent_type, max_dir)
    
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
def merge(old, new):

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
#Code Generation Algorithms
###############################################################################
def write_dict_directory(dct_data, indent_type='\t', max_dir=3, _level=()):
    code  = ''
    go_up = False
    
    for key, value in dct_data.items():
        key = format_key(key)
        
        if hasattr(value, 'to_dunl') and ut.isdictlike(value):
            chunk = value.to_dunl()
            
            directory  = ['']*len(_level) + [key]
            directory  = write_directory(directory)
            code      += directory + '\n' + chunk +  '\n'
            
            go_up = True
        
        elif ut.isdictlike(value):
            if max_dir:
                chunk = write_dict_directory(value, 
                                             max_dir=max_dir-1,
                                             indent_type=indent_type,
                                             _level=[*_level, key], 
                                             )
            else:
                chunk = write_dict(value, indent_type=indent_type)
                
            directory  = ['']*len(_level) + [key]
            directory  = write_directory(directory)
            code      += directory + '\n' + chunk +  '\n'
            
            go_up = True
        else:
            if not _level:
                raise ValueError('Insufficient nesting')
                
            value_  = write_value(value)
            line    = f'{key} : {value_}'
            
            if go_up:   
                directory  = write_directory(_level)
                code += directory + '\n' + line + '\n'
                go_up = False
            else:
                code += line + '\n'
    
    return code.strip() + '\n'

def write_directory(directory_lst):
    return ''.join([f';{x}' for x in directory_lst])

def write_dict(dct_data, multiline=True, indent_type='\t'):
    code = ''
    
    for key, value in dct_data.items():
        key = format_key(key)
        
        if hasattr(value, 'to_dunl') and ut.isdictlike(value):
            chunk = value.to_dunl()
            chunk  = tw.indent(chunk, indent_type)
            
            if multiline:
                code  += f'{key} : [\n' + chunk +  f'\n{indent_type}],\n'
            else:
                code  += f'{key} : [' + chunk +  '], '
            
        elif ut.isdictlike(value):
            chunk  = write_dict(value, multiline=multiline, indent_type=indent_type)
            chunk  = tw.indent(chunk, indent_type)
            
            if multiline:
                code  += f'{key} : [\n' + chunk +  f'\n{indent_type}],\n'
            else:
                code  += f'{key} : [' + chunk +  '], '
                
        else:
            value_  = write_value(value)
            line    = f'{key} : {value_}'
            
            if multiline:
                code   += line + ',\n'
            else:
                code += line + ', '
    
    if code[-2] == ',':
        code = code[:-2] 
    
    return code.strip()

def write_value(value, indent_type='\t'):
    '''Writes a primitive, list, dict or any object with the method `to_dunl`. 
    Not the same as a value in the readstring module.
    '''
    if hasattr(value, 'to_dunl'):
        return value.to_dunl(indent_type=indent_type)
    elif type(value) == bool:
        return str(value)
    elif ut.isnum(value) and not ut.isstrlike(value):
        return str(value)
    elif ut.isstrlike(value):
        return format_str(value)
    elif ut.islistlike(value):
        temp = [write_value(v) for v in value]
        return '[' + ', '.join(temp) + ']'
    elif ut.isdictlike(value):
        temp = write_dict(value, multiline=False)
        return '[' + temp.strip() + ']' 
    else:
        raise TypeError(f'Unexpected type of value : {value}')

def write_df(df, indent=1, indent_type='\t'):
    max_len_col = 0
    max_len_idx = 0
    columns     = []
    indices     = []
    for col in df.columns:
        if type(col) == tuple:
            col = [str(i) for i in col]
        else:
            col = str(col)

        max_len_col = max(len(col), max_len_col)
        columns.append(col)
        
    for idx in df.index:
        if type(idx) == tuple:
            idx = [str(i) for i in idx]
        else:
            idx = str(idx)
        
        max_len_idx = max(len(idx), max_len_idx)
        indices.append(idx)
        
        

###############################################################################
#Low Level Utilities
###############################################################################
def isprimitive(v):
    if not ut.isnum(v) and not ut.isstrlike(v) and not type(v) == bool:
        return False
    else:
        return True

def format_key(key):
    #Need to work on this
    formatted = str(key)
    
    return formatted

def format_str(string):
    string = str(string)
    if needs_quotes(string):
        formatted = "'" + string + "'"
        try:
            f'x = "{formatted}"'
            return formatted
        except:
            raise ValueError('Could not format string. Check the quotation marks.')
    else:
        return string

special_characters = [';', '$', '#', "'", '"']

def needs_quotes(s: str) -> bool:
    if s in ['True', 'False']:
        return True
    
    if len(s) > 1:
        if s[0] == s[-1] and s[0] in ["'", '"']:
            return False
        
    for c in special_characters:
        if c in s:
            return True
    
    if ut.isnum(s):
        return True
    return False

