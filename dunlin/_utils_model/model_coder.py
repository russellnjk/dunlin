import re

###############################################################################
#Supporting Imports
###############################################################################
import numpy          as np
import pandas         as pd
import scipy.optimize as opt
from   numba  import jit 
#Do NOT remove these imports. They are needed to run dynamically compiled functions.

###############################################################################
#Function Generation
###############################################################################
def model2func(model_dict, *args, **kwargs):
    code      = model2code(model_dict, *args, **kwargs)
    func_name = 'model_' + model_dict['name']
    exec(code)
    
    return locals()[func_name], code

def exvs2func(model_dict, exvs, *args, **kwargs):
    code  = exvs2code(model_dict, exvs, *args, **kwargs)
    funcs = {}
    
    exec(code, None, )

    for exv_name in exvs:
        func_name       = 'exv_{}_{}'.format(model_dict['name'], exv_name)
        funcs[exv_name] = locals()[func_name]

    return funcs, code

def modify2func(model_dict, modifier, *args, **kwargs):
    code      = modify2code(model_dict, modifier, *args, **kwargs)
    
    exec(code)
    
    func_name = 'modify_' + model_dict['name']
    
    return locals()[func_name], code
    
###############################################################################
#High-level Code Generation
###############################################################################
def model2code(model_dict, filename='', include_imports=True):
    use_numba  = '@use_numba' in model_dict['equations']
    imports    = 'import numpy as np\nfrom numba import jit' if use_numba else 'import numpy as np'
    result     = imports if include_imports else ''
    func_name  = 'model_' + model_dict['name']
    func_args  = '(t, y, p, u):' if model_dict['inputs'] else '(t, y, p):'
    func_def   = '@jit(nopython=True)\ndef '    if use_numba            else 'def ' 
    func_def   = func_def + func_name + func_args
    states     = vars2code('y', model_dict['states'])
    params     = vars2code('p', model_dict['params'])
    equations  = equations2code(model_dict['equations'].replace('@use_numba', ''), model_dict['states'], indent=1)

    if model_dict['inputs']:
        inputs     = vars2code('u', model_dict['inputs'])
        result     = '\n\n'.join([result, func_def, states, params, inputs, equations])
    else:
        result     = '\n\n'.join([result, func_def, states, params, equations])

    if filename:
        with open(filename, 'w') as file:
            file.write(result)
    return result

def exvs2code(model_dict, exvs, filename='', include_imports=True):
    states        = vars2code('y', model_dict['states'])
    params        = vars2code('p', model_dict['params'])
    inputs        = vars2code('u', model_dict['inputs']) if model_dict['inputs'] else ''
    eqns_template = equations2code(model_dict['equations'].replace('@use_numba', ''), indent=1)
    result        = ''
    import_numba  = False
    
    #Create code for each function
    for exv_name, exv_eqns in exvs.items():
        use_numba  = '@use_numba'  in exv_eqns 
        func_name  = 'exv_{}_{}'.format(model_dict['name'], exv_name)
        func_args  = '(t, y, p, u):' if model_dict['inputs'] else '(t, y, p):'
        func_def   = '@jit(nopython=True)\ndef ' if use_numba else 'def '
        func_def   = func_def + func_name + func_args
        exv_eqns_  = exv_eqns.replace('@use_numba', '')
        equations  = exv_equations2code(exv_eqns_, eqns_template, indent=1)

        if model_dict['inputs']:
            result = '\n\n'.join([result, func_def, states, params, inputs, equations])
        else:
            result = '\n\n'.join([result, func_def, states, params, equations])
        
        if use_numba:
            import_numba = True
    
    #Add import statements if required
    if import_numba and include_imports:
        result = 'import numpy as np\nfrom numba import jit\n\n' + result
    elif include_imports:
        result = 'import numpy as np\n\n' + result
        
    if filename:
        with open(filename, 'w') as file:
            file.write(result)
    return result

def modify2code(model_dict, modifier, filename='', include_imports=True):
    imports    = 'import numpy  as np'
    result     = imports if include_imports else ''
    func_name  = 'modify_' + model_dict['name']
    states     = vars2code('init', model_dict['states'])
    params     = vars2code('params', model_dict['params'])
    inputs     = vars2code('inputs', model_dict['inputs']) if model_dict['inputs'] else ''
    
    #Return values
    return_init   = '\tnew_init   = np.array([{}])'.format(', '.join(model_dict['states']))
    return_params = '\tnew_params = np.array([{}])'.format(', '.join(model_dict['params']))
    return_inputs = '\tnew_inputs = np.array([{}])'.format(', '.join(model_dict['inputs'])) if model_dict['inputs'] else ''
    
    if model_dict['inputs']:
        func_def   = 'def ' + func_name + '(function, init, params, inputs, scenario, segment):' 
        equations  = modify_equations2code(modifier)
        return_val = '\treturn new_init, new_params, new_inputs'
        result     = '\n\n'.join([result, func_def, states, params, inputs, equations, return_init, return_params, return_inputs, return_val])
    else: 
        func_def   = 'def ' + func_name + '(function, init, params, scenario, segment):'
        equations  = modify_equations2code(modifier)
        return_val = '\treturn new_init, new_params'
        result     = '\n\n'.join([result, func_def, states, params, equations, return_init, return_params, return_val])
    
    return result
    
###############################################################################
#Code Generation for Modifier
###############################################################################     
def modify_equations2code(mod_eqns, indent=1): 
    # equations_ = re.sub('@short .*(\n?:.*)+', lambda m: parse_short(m[0]),      equations )
    # equations_ = re.sub('(\|+)(.*)', lambda m: parse_tabs(m[1], m[2]), equations_)    
    equations_ = parse_shorthand(mod_eqns)
    
    result = '\n'.join( [t(indent) + line for line in equations_.split('\n')] )
    return result

def modify_vars2code(name, lst, indent=1):
    longest = len(max(lst, key=len))
    temp    = ["{}{}{}= {}[{}]".format(t(indent), x, sp(longest, x), name, i) for i, x in enumerate(lst)]
    result  = '\n'.join(temp)
    
    return result
    
###############################################################################
#Code Generation for exv Functions
###############################################################################    
def exv_equations2code(exv_eqns, eqns_template, indent=1):
    # equations_ = re.sub('@short .*(\n?:.*)+', lambda m: parse_short(m[0]),      exv_eqns  )
    # equations_ = re.sub('(\|+)(.*)', lambda m: parse_tabs(m[1], m[2]), equations_)   
    equations_ = parse_shorthand(exv_eqns)
    
    result = '\n'.join([t(indent) + line for line in equations_.split('\n')])
    result = '\n\n'.join([eqns_template, result])
    
    #Remove np.min, np.max
    return result

###############################################################################
#Code Generation for Numerical Integration
###############################################################################
def vars2code(name, lst, indent=1):
    longest = len(max(lst, key=len))
    temp    = ["{}{}{}= {}[{}]".format(t(indent), x, sp(longest, x), name, i) for i, x in enumerate(lst)]
    result  = '\n'.join(temp)
    
    return result
    
def equations2code(equations, states=None, indent=1):
    # equations_ = re.sub('@short .*(\n?:.*)+', lambda m: parse_short(m[0]),      equations )
    # equations_ = re.sub('(\|+)(.*)', lambda m: parse_tabs(m[1], m[2]), equations_)   
    equations_ = parse_shorthand(equations)
    
    result = '\n'.join([t(indent) + line for line in equations_.split('\n')])
    
    if states:
        diff         = ['d'+state for state in states]
        return_value = '{}return np.array([{}])'.format(t(indent), ', '.join(diff))
        result       = '\n\n'.join([result, return_value])    
    
    return result
        
###############################################################################
#Shorthand Coding
###############################################################################
def parse_shorthand(equations):
    equations_ = re.sub('@short .*(\n?:.*)+', lambda m: parse_short(m[0]),      equations )
    equations_ = re.sub('@lst .*(\n?:.*)+',   lambda m: parse_short(m[0]),      equations_)
    equations_ = re.sub('@arr .*(\n?:.*)+',   lambda m: parse_short(m[0]),      equations_)
    equations_ = re.sub('(\|+)(.*)',          lambda m: parse_tabs(m[1], m[2]), equations_)   
    
    return equations_
    
def parse_short(string):

    string_ = split_top_level(string[7:], delimiter=':')
    values  = [[v.strip() for v in split_top_level(vals)] for vals in string_[1:]]
    zipped  = zip(*values)
    expr    = string_[0].strip()
    
    result = '\n'.join([expr.format(*args) for args in zipped])
    
    null_pattern = '[\d\w\+\-\*\/\.\&\|]*\^[\d\w\+\-\*\/\.\&\|]*'
    result       = re.sub(null_pattern, lambda m:  replace_null(m[0]), result)
    
    return result

def replace_null(string):
    return ' '*len(string)

def parse_lst(string):
    
    string_ = split_top_level(string[5:], delimiter=':')
    values  = [[v.strip() for v in split_top_level(vals)] for vals in string_[1:]]
    zipped  = zip(*values)
    expr    = string_[0].strip()
    
    result = ', '.join([expr.format(*args) for args in zipped])
    
    return '[' + result + ']'

def parse_arr(string):
    lst = parse_lst(string)
    return f'np.array({lst})'

def parse_tabs(pipes, expr):
    return pipes.replace('|', '\t') + expr.strip()

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

###############################################################################
#Supporting Functions
###############################################################################
def sp(longest, state):
    return ' '*(longest - len(state) + 1)

def t(indent):
    return '\t'*indent

###############################################################################
#Removing Inner Function Calls
###############################################################################


if __name__ == '__main__':
    
    __model__ = {'name'     : 'testmodel', 
                 'states'   : ['x', 's', 'h'], 
                 'params'   : ['ks', 'mu_max', 'synh', 'ys'], 
                 'inputs'   : ['b'], 
                 'equations': '\nmu = mu_max*s/(s+ks)\n\n#Differentials\ndx = mu*x\nds = -dx/ys + b\ndh = synh - h*mu', 
                 'meta'     : {}
                 }
    # __model__['equations'] += '\n@short d{} =-k*{}: x1, x2, x3: x1, x2, x3'
    # #Not part of the main tests
    # states    = vars2code(list(__model__['states']))
    # params    = vars2code(list(__model__['params']))
    # inputs    = vars2code(list(__model__['inputs']))
    # equations = equations2code(__model__['equations'], list(__model__['states']), indent=1)
    
    # #code generation
    # code = model2code(__model__)
    # print(code)
    
    # exvs = {'exv_1': 'return t, -dx/ys',
    #               'exv_2': 'return t, mu'
    #               }
    
    # states    = exv_vars2code(list(__model__['states']))
    # params    = exv_vars2code(list(__model__['params']))
    # inputs    = exv_vars2code(list(__model__['inputs']))
    
    # eqns_template = equations2code(__model__['equations'], indent=1)
    # equations     = exv_equations2code(exv_eqns, eqns_template, indent=1)
    
    # code = exvs2code(__model__, exvs)
    
    #Test model function
    func, code = model2func(__model__)
    
    y0     = np.array([1, 1, 1])
    t0     = 0.0
    params = np.array([1, 1, 1, 1])
    inputs = np.array([0])
    
    dy = func(t0, y0, params, inputs)
    
    assert all(dy == [0.5, -0.5, 0.5])
    
    exvs = {'exv_1': 'return t, -dx/ys',
            'exv_2': 'return t, mu'
            }
    
    #Test exv functions
    funcs, code = exvs2func(__model__, exvs)
    
    y1     = np.array([[1, 1.5], [1, 0.5], [1, 1.5]])
    t1     = np.array([0, 1])
    p1 = np.array([[1, 1],[ 1, 1], [1, 1], [1, 1]])
    u1 = np.array([[0, 0]])
    
    
    xo1, yo1 = funcs['exv_1'](t1, y1, p1, u1)
    assert all(xo1 == [0, 1])
    assert all(yo1 == [-0.5, -0.5])
    
    xo2, yo2 = funcs['exv_2'](t1, y1, p1, u1)
    assert all(xo2 == [0, 1])
    assert all(np.isclose(yo2, [0.5, 0.33333]))
    
    #Test shorthand
    __model__ = {'name'     : 'testmodel', 
                 'states'   : ['x', 's', 'h'], 
                 'params'   : ['ks', 'mu_max', 'synh', 'ys'], 
                 'inputs'   : ['b'], 
                 'equations': '\nmu = mu_max*s/(s+ks)\n\n#Differentials\ndx = mu*x\nds = -dx/ys + b\n@short d{} = syn{} -{}*mu: h: h: h', 
                 'meta'     : {}
                 }
    
    exv, code = model2func(__model__)
    
    y0     = np.array([1, 1, 1])
    t0     = 0.0
    params = np.array([1, 1, 1, 1])
    inputs = np.array([0])

    dy = exv(t0, y0, params, inputs)
    
    assert all(dy == [0.5, -0.5, 0.5])
    
    #Test modifier function
    modifier   = 'if segment == 0:\n|x = 0'
    model_func = lambda x: x
    
    mod, mod_code = modify2func(__model__, modifier, model_func)
    
    new_init, new_params, new_inputs = mod(func, y0, params, inputs, 0, 0)    
    
    assert all(new_init == [0, 1, 1])
    assert all(new_params == params)
    assert all(new_inputs == inputs)