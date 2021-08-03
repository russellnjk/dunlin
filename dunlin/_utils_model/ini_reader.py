import numpy   as np
from   numpy   import linspace
from   pathlib import Path

###############################################################################
#Non-Standard Imports
###############################################################################
try:
    from .custom_eval import safe_eval as eval
except Exception as e:
    if Path.cwd() == Path(__file__).parent:
        from custom_eval import safe_eval as eval
    else:
        raise e
        
###############################################################################
#Globals
###############################################################################
in_model = set(['states',      'params',   'inputs', 
                'equations',   'tspan',    'meta', 
                'solver_args', 'modify_eqns'
                ])
to_split = set(['exv_'])

###############################################################################
#Main Algorithm
###############################################################################
def parse_section(model_name, config, append_model_name=False):
    #Overhead
    global all_
    in_model = all_['in_model']
    data     = dict(config[model_name])
    model    = {'name'              : model_name, 
                'append_model_name' : append_model_name
                }
    
    #Create the dict to be returned
    section_data = {'model'       : model,
                    'ini_section' : data,
                    }
    
    for key in data:
        #Get the function
        key_ = 'exv' if key[:4] == 'exv_' else key
            
        try:
            func = all_['parse_' + key_]
        except:
            raise Exception(f'No function for parsing {key}')
        
        #Call the function
        try:
            string       = data[key]
            parsed_value = func(string, append_model_name)
        except Exception as e:
            msg = f'Error in parsing {key} when reading .ini file.\n{string}\n'
            combine_error_msg(msg, e)
            raise e
        
        #Place the results in the correct location
        if key in in_model:
            section_data['model'][key] = parsed_value
        elif key[:4] == 'exv_':
            section_data['model'].setdefault('exv_eqns', {})
            section_data['model']['exv_eqns'][key[4:]] = parsed_value
        else:
            section_data[key] = parsed_value
            
    return section_data

def combine_error_msg(msg, e):
    args   = (msg,) + e.args
    args   = '\n'.join(args)
    e.args = (args, )
    return e

###############################################################################
#Wrappers and Checkers
###############################################################################
def is_listlike(x):
    '''
    :meta private:
    '''
    try:
        list(x)
        return True
    except:
        return False
    
def try_int(x):
    '''
    Attempts to convert a str into an int.
    :meta private:
    '''
    try:
        return int(x)
    except:
        return x

###############################################################################
#Subsection Parsers
###############################################################################
'''
Rules for parsers
1. The function name is parse_<key>
2. The signature is: string, model_name='', append_model_name=False
3. Add the key to the global variable in_model if the key is a model attribute 
    and does not require splitting or "special" processing
4. Modify the loop in parse_section if it does.
'''
def parse_states(string, model_name='', append_model_name=False):
    return string2dict(string)

def parse_params(string, model_name='', append_model_name=False):
    result = string2dict(string)
    if append_model_name:
        result = append_name_to_keys(model_name, result)
    return result

def parse_inputs(string, model_name='', append_model_name=False):
    if not string:
        return {}
    
    s = string.replace('\n', '')
    s = s.replace('\t', '')
    
    pairs = [[i.strip() for i in pair.split('=')] for pair in split_top_level(s)]
    pairs = {(pair[0], seg): v for pair in pairs for seg, v in enumerate(eval(pair[1])) }
        
    return pairs

def parse_equations(string, model_name='', append_model_name=False):
    return string

def parse_tspan(string, model_name='', append_model_name=False):
    s     = string.strip()
    tspan = eval(s)
        
    return tspan

def parse_solver_args(string, model_name='', append_model_name=False):
    return string2dict(string)

def parse_meta(string, model_name='', append_model_name=False):
    return string2dict(string)

def parse_exv(string, model_name='', append_model_name=False):
    return string

def parse_modify(string, model_name='', append_model_name=False):
    return string

def parse_priors(string, model_name='', append_model_name=False):
    result = string2dict(string, set_len=2)
    if append_model_name:
        result = append_name_to_keys(model_name, result)
    return result

def parse_param_bounds(string, model_name='', append_model_name=False):
    result = string2dict(string, set_len=2)
    if append_model_name:
        result = append_name_to_keys(model_name, result)
    return result

def parse_step_size(string, model_name='', append_model_name=False):
    result = string2dict(string, set_len=0)
    if append_model_name:
        result = append_name_to_keys(model_name, result)
    return result

def parse_iterations(string, model_name='', append_model_name=False):
    iterations = int(string)
    return iterations

def parse_combinations(string, model_name='', append_model_name=False):
    combinations_dict = parse_params(string)
    return combinations_dict

def parse_combination_spacing(string, model_name='', append_model_name=False):
    return string

def parse_measured_states(string, model_name='', append_model_name=False):
    return eval(string)

def parse_cf_iterations(string, model_name='', append_model_name=False):
    return eval(string)

###############################################################################
#Low-Level String Operations
###############################################################################
def split_top_level(string, delimiter=','):
    result = []
    start  = 0 
    level  = 0
    
    for index, char in enumerate(string):
        if char == delimiter and level == 0:
            result.append(string[start: index])
            start = index + 1
        elif char in ['(', '[', '{']:
            level += 1
        elif char in [')', ']', '}']:
            level -= 1
        else:
            pass
    result.append(string[start:])
    return result

def string2dict(string, set_len=None, allow_dictlike=False, max_dict_depth=1):
    
    def check_value_len(value, set_len, allow_dictlike):
        if set_len is None:
            return value
        elif type(value) == dict:
            if not allow_dictlike:
                raise TypeError('Dictlike values not allowed for this string.')
            elif len(value) != set_len:
                raise Exception(f'Expected value of length {set_len}')
            # elif any([is_listlike(v) for v in value.values()]):
            #     msg = f'Expected pair-like substring. Detected {value}'
            #     raise ValueError(msg)
            return value
        elif is_listlike(value):
            if set_len == 0:
                raise Exception('Only point values are allowed for this string.')
            elif len(value) != set_len:
                raise Exception(f'Expected value of length {set_len}')
            return list(value)
        else:
            return value
            
    def helper(string, set_len, allow_dictlike, depth=0):
        result = {}
        for pair in split_top_level(string):
            pair_ = pair.strip()
            if not pair_:
                continue
            
            pair_      = split_top_level(pair_, '=')
            if len(pair_) != 2:
                msg = f'Expected pair-like substring. Detected {pair}'
                raise ValueError(msg)
                
            key, value = pair_
            
            key_   = key.strip()
            
            try:
                value_ = eval(value)
                value_ = check_value_len(value_, set_len, allow_dictlike)
            except SyntaxError as e:
                if depth < max_dict_depth:
                    value_ = value.strip()
                    value_ = value_[1:len(value_)-1]
                    value_ = helper(value_, set_len=1, allow_dictlike=False, depth=depth+1)
                else:
                    raise e
            except Exception as e:
                raise e
            result[key_] = value_
        
        return result
    
    string_ = string.replace('\n', '')
    string_ = string_.replace('\t', '')
    
    return helper(string_, set_len, allow_dictlike)

# def string2dict(string, set_len=None):
#     if not string:
#         return {}
    
#     s = string.replace('\n', '')
#     s = s.replace('\t', '')
    
#     pairs = [[i.strip() for i in pair.split('=')] for pair in split_top_level(s)]
#     # pairs = dict([[pair[0], eval(pair[1]) ] for pair in pairs])
#     result = {}
#     for key, value in pairs:
#         result[key] = eval(value)
        
#         if set_len is None:
#             continue
#         elif set_len == 0 and is_listlike(result[key]):
#             raise Exception('List-like values not allowed.')
#         elif len(result[key]) != set_len:
#             raise Exception(f'Expected list-like value of length {set_len}')
            
#     return result

def append_name_to_keys(name, dict_data):
    return {name + '_' + key: value for key, value in dict_data.items()}

###############################################################################
#Caching
###############################################################################
all_ = globals()

if __name__ == '__main__':
    t = 'linspace(0, 30, 31), linspace(30, 60, 31)'
    
    tspan = parse_tspan(t)
    
    p = 'x = [1, 2],\ny = [3]*2'
    
    params = parse_params(p)