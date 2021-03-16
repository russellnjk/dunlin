import numpy        as np

###############################################################################
#Non-Standard Imports
###############################################################################
from .utils_eval import safe_eval as eval

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
def parse_section(name, config):
    data = config[name]
    
    model, tspan, init_vals, params_vals, inputs_vals = parse_model_args(name, config)
    
    section_data  = {'model'   : model,
                     }
    
    #Objective functions for simulation
    objs = parse_objective_args(name, config)
    if objs:
        section_data['objectives'] = objs 
        
    for key in data:
        #Priors
        if key == 'priors':
            section_data['priors'] = parse_priors(name, config)
        
        #Parameter bounds
        if key == 'param_bounds':
            section_data['param_bounds'] = parse_bounds(name, config, 'param_bounds')
        
        #Step size
        if key == 'step_size':
            section_data['step_size'] = parse_step_size(name, config)
        
        #Curve-fitting iterations
        if key == 'cf_iterations':
            section_data['cf_iterations'] = parse_iterations(name, config, 'cf_iterations')
        
        #Combinations
        if key == 'combinations':
            section_data['combinations'] = parse_combinations(name, config)
            
        #Combination spacing
        if key == 'combination_spacing':
            section_data['combination_spacing'] = parse_combination_spacing(name, config)
        
        #Measured states
        #Decomposition
    
    return section_data
        
###############################################################################
#Subsection Parsers
###############################################################################
@wrap_try('model arguments')
def parse_model_args(name, config):
    data = config[name]
    
    #Process tspan
    tspan = parse_tspan(data['tspan'])
        
    #Process int_args 
    init_dict   = parse_init(data['states'])
    params_dict = parse_params(data['params'])
    
    if 'inputs' in data:
        inputs_mode = data.get('input_mode', 'scenario')
        inputs_dict = parse_inputs(data['inputs'])
    else:
        inputs_dict = None

    #Process solver_args and other stuff
    solver_args = parse_solver_args(data.get('solver_args', {}))
    use_numba   = data.get('use_numba', True)
    
    #Compile model args
    model = {'name'        : name,
             'states'      : init_dict,
             'params'      : params_dict,
             'inputs'      : inputs_dict,
             'meta'        : parse_meta(data.get('meta')),
             'equations'   : data['equations'],
             'tspan'       : tspan,
             'solver_args' : solver_args,
             'use_numba'   : use_numba
             }
    
    return model, tspan, init_dict, params_dict, inputs_dict

@wrap_try('objective function')
def parse_objective_args(name, config):
    data = config[name]
    
    #Process objectives
    objs = {}
    for key in data:
        if 'objective_' in key:
            obj_key  = key.split('objective_')[1]
            obj_eqns = data[key]
            
            objs[obj_key] = obj_eqns
    
    return objs

@wrap_try('fixed_params')
def parse_fixed_params(name, config):
    data = config[name]
    
    temp         = data['fixed_params'].replace('[', '').replace(']', '').split(',')
    fixed_params = [s.strip() for s in temp if s.strip()]
    
    return fixed_params

@wrap_try('priors')
def parse_priors(name, config):
    data = config[name]
    
    priors_dict = parse_params(data['priors'])
    return priors_dict

@wrap_try('bounds')
def parse_bounds(name, config, key='bounds'):
    data = config[name]
    
    bounds_dict = parse_params(data['param_bounds'])
    return bounds_dict

@wrap_try('step_size')
def parse_step_size(name, config):
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
            
    return step_size_dict

@wrap_try('iterations')
def parse_iterations(name, config, key='iterations'):
    data = config[name]
    
    iterations = int(data[key])
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
    
    if s[0] in ['[', '(']:
        s = split_top_level(string)
    else:
        s = split_top_level(string)
        
    spans = [span.strip() for span in s]
    tspan = []
    
    for span in spans:
        if 'linspace' in span:
            args  = eval(span[8:])
            tspan.append(np.linspace(*args))
        else:
            tspan.append(eval(span))

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

if __name__ == '__main__':
    t = 'linspace(0, 30, 31), linspace(30, 60, 31)'
    
    tspan = parse_tspan(t)
    
    p = 'x = [1, 2],\ny = [3]*2'
    
    params = parse_params(p)