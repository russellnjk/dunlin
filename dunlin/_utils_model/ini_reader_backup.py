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
#Error Message Formatter
###############################################################################
def wrap_try(name):
    msg = 'Error in parsing {} when reading .ini file.'.format(name)
    def wrapper(func):
        def helper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                args   = (msg,) + e.args
                args   = '\n'.join(args)
                e.args = (args, )
                raise e
        return helper
    return wrapper

###############################################################################
#Main Algorithm
###############################################################################
def parse_section(name, config, append_model_name=False):
    data = dict(config[name])
    
    model, tspan, init_vals, params_vals, inputs_vals = parse_model_args(name, config, append_model_name)
    
    section_data = {'model'       : model,
                    'ini_section' : data,
                    }
    
    model['exv_eqns'] = {}
    for key in data:
        #exv
        if key[:4] == 'exv_':
            parse_exv_args(name, config, key, model['exv_eqns'])
            
        #Priors
        if key == 'priors':
            section_data['priors'] = parse_priors(name, config, append_model_name)
        
        #Parameter bounds
        if key == 'param_bounds':
            section_data['param_bounds'] = parse_param_bounds(name, config, append_model_name)
        
        #Step size
        if key == 'step_size':
            section_data['step_size'] = parse_step_size(name, config, append_model_name)
        
        #Curve-fitting iterations
        if key == 'cf_iterations':
            section_data['cf_iterations'] = parse_iterations(name, config)
        
        #Combinations
        if key == 'combinations':
            section_data['combinations'] = parse_combinations(name, config)
            
        #Combination spacing
        if key == 'combination_spacing':
            section_data['combination_spacing'] = parse_combination_spacing(name, config)
            
        #Measured states
        if key == 'measured_states':
            section_data['measured_states'] = parse_measured_states_args(name, config)
        
        #Decomposition
    return section_data
        
###############################################################################
#Subsection Parsers
###############################################################################
@wrap_try('model arguments')
def parse_model_args(name, config, append_model_name=False):
    data = config[name]
    
    #Process tspan
    tspan = parse_tspan(data['tspan'])
        
    #Process int_args 
    init_dict   = parse_init(data['states'])
    params_dict = parse_params(data['params'], append_model_name)
    
    if append_model_name:
        params_dict = append_name_to_keys(name, params_dict)
    
    if 'inputs' in data:
        inputs_dict = parse_inputs(data['inputs'])
    else:
        inputs_dict = None

    #Process solver_args and other stuff
    solver_args = parse_solver_args(data.get('solver_args', {}))
        
    #Compile model args
    model = {'name'        : name,
             'states'      : init_dict,
             'params'      : params_dict,
             'inputs'      : inputs_dict,
             'meta'        : parse_meta(data.get('meta')),
             'equations'   : data.get('equations'),
             'tspan'       : tspan,
             'solver_args' : solver_args,
             'modify_eqns' : data.get('modify')
             }
    
    return model, tspan, init_dict, params_dict, inputs_dict

@wrap_try('extra variable function')
def parse_exv_args(name, config, key, exvs):
    data = config[name]
    
    #Process exvs
    exv_key       = key.split('exv_')[1]
    exv_eqns      = data[key]
    exvs[exv_key] = exv_eqns
    
    return exvs

@wrap_try('modifier function')
def parse_modify_args(name, config):
    data = config[name]
    
    modify = data['modify']
    
    return modify

@wrap_try('priors')
def parse_priors(name, config, append_model_name=False):
    data = config[name]
    
    priors_dict = parse_params(data['priors'])
    
    if append_model_name:
        priors_dict = append_name_to_keys(name, priors_dict)
    return priors_dict

@wrap_try('param_bounds')
def parse_param_bounds(name, config, append_model_name=False):
    data = config[name]
    
    bounds_dict = parse_params(data['param_bounds'])
    
    if append_model_name:
        bounds_dict = append_name_to_keys(name, bounds_dict)
    return bounds_dict

@wrap_try('step_size')
def parse_step_size(name, config, append_model_name=False):
    data = config[name]
    
    step_size_dict = parse_params(data['step_size'])
    for key, value in list(step_size_dict.items()):
        if type(value) == list:
            if len(value) > 1:
                msg = 'step_size can only have one value per parameter. Received {} values for {}: {}'
                raise ValueError(msg.format(len(value), key, value))
            else:
                step_size_dict[key] = float(value[0])
        else:
            step_size_dict[key] = float(value)
    
    if append_model_name:
        step_size_dict = append_name_to_keys(name, step_size_dict)
        
    return step_size_dict

@wrap_try('cf_iterations')
def parse_iterations(name, config):
    data = config[name]
    
    iterations = int(data['cf_iterations'])
    return iterations

@wrap_try('combinations')
def parse_combinations(name, config):
    data = config[name]
    try:
        int(data['combinations'])
    except:
        combinations_dict = parse_params(data['combinations'])
    return combinations_dict

@wrap_try('combination_spacing')
def parse_combination_spacing(name, config):
    data = config[name]
    
    combination_spacing = data['combination_spacing']
    return combination_spacing

@wrap_try('measured_states')
def parse_measured_states_args(name, config):
    data = config[name]
    
    return eval(data['measured_states'])
    
###############################################################################
#Supporting Functions
###############################################################################
def try_int(x):
    '''
    Attempts to convert a str into an int.
    
    :meta private:
    '''
    try:
        return int(x)
    except:
        return x

def try_str(func):
    '''
    Creates a custom exception message containing the original string argument.
    
    :meta private:
    '''
    def helper(string, *args, **kwargs):
        try:
            return func(string)
        except Exception as e:
            msg = 'Error in parsing this string: \n{}'.format(string)
            msg = '\n'.join([msg] + [str(x) for x in e.args])
            raise Exception(msg)
    return helper        
    
###############################################################################
#Low-Level .ini Parsing
###############################################################################
@try_str
def parse_tspan(string):
    s = string.strip()
    tspan = eval(s)
        
    return tspan

@try_str
def parse_inputs(string):
    if not string:
        return {}
    
    s = string.replace('\n', '')
    s = s.replace('\t', '')
    
    pairs = [[i.strip() for i in pair.split('=')] for pair in split_top_level(s)]
    pairs = {(pair[0], seg): v for pair in pairs for seg, v in enumerate(eval(pair[1])) }
        
    return pairs

@try_str
def parse_init(string):
    return string2dict(string)

@try_str
def parse_params(string):
    return string2dict(string)

@try_str
def parse_solver_args(string):
    return string2dict(string)

@try_str
def parse_meta(string):
    return string2dict(string)

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

def string2dict(string):
    if not string:
        return {}
    
    s = string.replace('\n', '')
    s = s.replace('\t', '')
    
    pairs = [[i.strip() for i in pair.split('=')] for pair in split_top_level(s)]
    pairs = dict([[pair[0], eval(pair[1]) ] for pair in pairs])
    
    return pairs

def append_name_to_keys(name, dict_data):
    return {name + '_' + key: value for key, value in dict_data.items()}

if __name__ == '__main__':
    t = 'linspace(0, 30, 31), linspace(30, 60, 31)'
    
    tspan = parse_tspan(t)
    
    p = 'x = [1, 2],\ny = [3]*2'
    
    params = parse_params(p)