import configparser as cp
import numpy        as np
import pandas       as pd

###############################################################################
#Extra
###############################################################################
def get_ini_args(model_data):
    '''
    :meta private:
    '''
    return {key: value['ini_section'] for key, value in model_data.items()}

###############################################################################
#Model to .ini
###############################################################################
def update_ini_sections(ini_sections, is_model_data=False, inplace=False, filename='', **to_update):
    config = cp.ConfigParser()
    config.optionxform = str 
    
    for model_key, value in ini_sections.items():
        ini_section = value['ini_section'] if is_model_data else value
        to_update_  = to_update.get(model_key, {})
        new_section = update_ini_section(ini_section = ini_section, 
                                         inplace     = inplace, 
                                         **to_update_
                                         )
        config[model_key] = new_section
        
        if inplace:
            value['ini_section'] = new_section
    
    if filename:
        with open(filename, 'w') as file:
            config.write(file)
            
    return config
    
def update_ini_section(ini_section=None, inplace=False, **to_update):
    #Parse 
    funcs = {'states': init2str,
             'params': params2str,
             'inputs': inputs2str,
             'tspan' : tspan2str,
             }
    temp  = {}

    for key, value in to_update.items():

        try:
            func = funcs[key]
        except:
            raise Exception(f'No function for parsing {key}')
        
        try:
            temp[key] = func(value)
        except:
            raise Exception(f'Could not update the following key and value. key: {key}\nvalue: {value}')
        
    #Merge with ini_section is provided
    if ini_section:
        new_section = {**ini_section, **temp}
    else:
        new_section = temp
    
    #Update if inplace
    if inplace:
        ini_section = new_section
    
    return new_section
        
###############################################################################
#Integration Arguments to str
###############################################################################
def inputs2str(input_vals, no_subsection=True, **numargs):
    
    indent       = 0 if no_subsection else 1 
    aligned_keys = align_keys(input_vals)
    collated     = {}
    for segment, df_ in input_vals.groupby(level=1):
        
        aligned_dict = column2list(df_, **numargs)
    
        for variable, aligned_lst in aligned_dict.items():
            if variable in collated:
                collated[variable] += ', ' + aligned2str(aligned_lst) 
            else:
                collated[variable] = aligned2str(aligned_lst) 
    
    result = [make_line(indent, aligned_keys[key], '[' + value + ']')  for key, value in collated.items()]
    result = '\n' + ',\n'.join(result) if no_subsection else 'inputs = \n' + ',\n'.join(result)
    
    return result

def init2str(init_vals, no_subsection=True, **numargs):
    indent       = 0 if no_subsection else 1 
    aligned_keys = align_keys(init_vals)
    
    aligned_dict = column2list(init_vals, **numargs)
    
    result = [make_line(indent, aligned_keys[key], '[' + ', '.join(value) + ']')  for key, value in aligned_dict.items()]
    result = '\n' + ',\n'.join(result) if no_subsection else 'states = \n' + ',\n'.join(result)
    
    return result

def params2str(param_vals, no_subsection=True, **numargs):
    indent       = 0 if no_subsection else 1 
    aligned_keys = align_keys(param_vals)
    
    aligned_dict = column2list(param_vals, **numargs)
    
    result = [make_line(indent, aligned_keys[key], '[' + ', '.join(value) + ']')  for key, value in aligned_dict.items()]
    result = '\n' + ',\n'.join(result) if no_subsection else 'params = \n' + ',\n'.join(result)
    
    return result

def tspan2str(tspan, no_subsection=True, **numargs):
    result = []
    for tseg in tspan:
        lst          = [num2sci(tpoint, **numargs) for tpoint in tseg]
        tseg_string  = '[' + ', '.join(lst) + ']'
        result.append(tseg_string)
    
    result = '\n[' + ', '.join(result) + ']' if no_subsection else 'tspan = \n\t[' + ' ,'.join(result) + ']' 
    return result
    
###############################################################################
#Dict2String Supporting Functions
###############################################################################
def align_spaces(lst, n_spaces_lst):
    '''
    :meta private:
    '''
    return [string + ' ' * n for string, n in zip(lst, n_spaces_lst)]
    
def column2list(df, **numargs):
    '''
    :meta private:
    '''
    wrap_num  = lambda x: num2sci(x, **numargs)
    df_string = df.applymap(wrap_num)
    lengths   = df_string.applymap(len)
    max_len   = lengths.max(axis=1)
    n_spaces  = lengths.apply(lambda x: max_len.values-x , axis=0)

    aligned_dict = {}
    for variable, lst in df_string.items():
        aligned_lst = [string + ' ' * n for string, n in zip(lst, n_spaces[variable])]
            
        aligned_dict[variable] = aligned_lst
            
        pass
    return aligned_dict

def aligned2str(aligned_lst):
    '''
    :meta private:
    '''
    return '[' + ', '.join(aligned_lst) + ']'

def align_keys(df):
    '''
    :meta private:
    '''
    longest      = len(max(df.keys(), key=len))
    aligned_keys = {}
    for key in df.keys():
        aligned_keys[key] = key + ' '*(longest - len(key))
    return aligned_keys

def make_line(indent, key, value):
    '''
    :meta private:
    '''
    return '\t'*indent + key + ' = ' + value

###############################################################################
#Low-Level String Conversion
###############################################################################
def num2sci(num, lb=-1, ub=3):
    '''
    :meta private:
    '''
    if num == 0:
        return '0'
    elif num < 10**lb or num > 10**ub:
        return '{:.2e}'.format(num).replace('e+0', 'e').replace('e-0', 'e-')
    else:
        return str(round(num))

if __name__ == '__main__':
    pass