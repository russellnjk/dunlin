import configparser as cp
import numpy        as np
import pandas       as pd
import textwrap     as tw

###############################################################################
#Model data to .ini
###############################################################################
def make_updated_inicode(sections, is_model_data=True, filename='', has_model_name=False, **to_update):
    new_sections = {}
    
    for model_name, value in sections.items():
        section     = value['ini_section'] if is_model_data else value
        to_update_  = to_update.get(model_name, {})
        new_section = make_updated_section(section        = section, 
                                           has_model_name = has_model_name,
                                           subsection     = True,
                                           **to_update_
                                          )
        new_sections[model_name] = '\n\n'.join(new_section.values())
    
    
    inicode = '\n\n'.join( [ f'[{model_name}]\n' + value for model_name, value in new_sections.items()] )
    if filename:
        with open(filename, 'w') as file:
            file.write(inicode)
            
    return inicode

def make_updated_config(sections, is_model_data=True, filename='', has_model_name=False, **to_update):
    config             = cp.ConfigParser()
    config.optionxform = str 
    
    for model_name, value in sections.items():
        section     = value['ini_section'] if is_model_data else value
        to_update_  = to_update.get(model_name, {})
        new_section = make_updated_section(section        = section, 
                                           has_model_name = has_model_name,
                                           subsection     = False,
                                           **to_update_
                                          )
        config[model_name] = new_section
    
    if filename:
        with open(filename, 'w') as file:
            config.write(file)
            
    return config
    
def make_updated_section(section=None, has_model_name=False, subsection=False, **to_update):
    global all_
    
    if section:
        if subsection:
            new_section = {key: make_subsection(key, value) for key, value in section.items()}
        else:
            new_section = section
    else:
        new_section = {}
    
    for key, value in to_update.items():
        #Get the function
        try:
            func = all_[key+'2str']
        except:
            raise Exception(f'No function for parsing {key}')
        
        #Call the function
        try:
            string = func(value, has_model_name=has_model_name)
        except Exception as e:
            msg = f'Could not update the following key and value. key: {key}\nvalue: {value}'
            combine_error_msg(msg, e)
            raise e
        
        #Format the strings
        if subsection:
            string = make_subsection(key, string)
        
        #Update new_section
        new_section[key] = string
    
    return new_section

def combine_error_msg(msg, e):
    args   = (msg,) + e.args
    args   = '\n'.join(args)
    e.args = (args, )
    return e

###############################################################################
#Wrapping Subsections
###############################################################################
def wrap_df(func):
    def helper(*args, **kwargs):
        
        if type(args[0]) == pd.DataFrame:
            df = args[0]
        elif type(args[0]) == pd.Series:
            df = pd.DataFrame(args[0]).T
        else:
            df = pd.DataFrame(args[0])
        result = func(df, *args[1:], **kwargs)
        
        return result
    return helper

def make_subsection(subsection_name, string):
    if '\n' in string:
        result = subsection_name + ' = \n' + tw.indent(string, '\t')
    else:
        result = subsection_name + ' = ' + string
    return result
        
###############################################################################
#String Writers
###############################################################################
'''
Rules for writers
1. The function name is <key>2str
2. The signature is: input_vals, has_model_name=False
3. Additional keyword arguments are allowed but will be ignored
'''
@wrap_df
def inputs2str(input_vals, has_model_name=False, **numargs):
    input_vals_  = remove_model_name(input_vals) if has_model_name else input_vals
    aligned_keys = align_keys(input_vals_)
    collated     = {}
    for segment, df_ in input_vals_.groupby(level=1):
        
        aligned_dict = column2list(df_, **numargs)
    
        for variable, aligned_lst in aligned_dict.items():
            if variable in collated:
                collated[variable] += ', ' + aligned2str(aligned_lst) 
            else:
                collated[variable] = aligned2str(aligned_lst) 
    
    result = [make_line(aligned_keys[key], '[' + value + ']')  for key, value in collated.items()]
    result = ',\n'.join(result)
    return result

@wrap_df
def states2str(init_vals, has_model_name=False,**numargs):
    init_vals_   = remove_model_name(init_vals) if has_model_name else init_vals
    aligned_keys = align_keys(init_vals_)
    
    aligned_dict = column2list(init_vals_, **numargs)
    
    result = [make_line(aligned_keys[key], '[' + ', '.join(value) + ']')  for key, value in aligned_dict.items()]
    result = ',\n'.join(result)
    return result

@wrap_df
def params2str(param_vals, has_model_name=False, **numargs):
    param_vals_  = remove_model_name(param_vals) if has_model_name else param_vals
    aligned_keys = align_keys(param_vals_)
    
    aligned_dict = column2list(param_vals_, **numargs)
    
    result = [make_line(aligned_keys[key], '[' + ', '.join(value) + ']')  for key, value in aligned_dict.items()]
    result = ',\n'.join(result)
    return result

def tspan2str(tspan, has_model_name=False, **numargs):
    result = []
    for tseg in tspan:
        lst          = [num2sci(tpoint, **numargs) for tpoint in tseg]
        tseg_string  = '[' + ', '.join(lst) + ']'
        result.append(tseg_string)
    
    result = '[' + ', '.join(result) + ']' 
    return result

def cf_iterations2str(cf_iterations, has_model_name=False):
    int_ = int(cf_iterations)
    if int_ != cf_iterations:
        raise ValueError('Expected an integer value for cf_iterations.')
    return str(int_)
    
###############################################################################
#Dict or DataFrame to String Supporting Functions
###############################################################################
def remove_model_name(df):
    '''
    :meta private:
    '''
    df_         = df.copy()
    df_.columns = [c.split('_', 1)[1] for c in df.columns]

    return df_
    
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

def make_line(key, value):
    '''
    :meta private:
    '''
    return key + ' = ' + value

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

###############################################################################
#Caching
###############################################################################
all_ = globals()

if __name__ == '__main__':
    idx = pd.MultiIndex.from_product([[0, 1], ['seg0', 'seg1']])
    df = pd.DataFrame([[1, 2], [100, 2e4], [3, 4], [300, 4e-4]], columns=['m1_u0', 'm1_u1'], index=idx)
    
    # r = inputs2str(df, has_model_name=True)
    # print(r)
    # r = init2str(df.xs(key='seg0', level=1))
    # print(r)
    
    # r = make_updated_section(inputs=df)
    # print(r)
    
    r = make_updated_inicode({'m1': {}}, m1={'inputs':df, 'cf_iterations': 3000}, is_model_data=False )
    print(r)
    
    r = make_updated_config({'m1': {}}, m1={'inputs':df, 'cf_iterations': 3000}, is_model_data=False  )
    print(r)
    
    