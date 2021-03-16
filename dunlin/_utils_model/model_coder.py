import re

###############################################################################
#Supporting Imports
###############################################################################
import numpy  as np
import pandas as pd
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

def objectives2func(model_dict, objectives, *args, **kwargs):
    code  = objectives2code(model_dict, objectives, *args, **kwargs)
    funcs = {}
    exec(code)

    for obj_name in objectives:
        func_name       = 'objective_{}_{}'.format(model_dict['name'], obj_name)
        funcs[obj_name] = locals()[func_name]

    return funcs, code
    
###############################################################################
#High-level Code Generation
###############################################################################
def model2code(model_dict, filename='', use_numba=True, include_imports=True):
    imports    = 'import numpy as np\nfrom numba import jit' if use_numba else 'import numpy as np'
    result     = imports if include_imports else ''
    func_name  = 'model_' + model_dict['name']
    func_args  = '(t, y, params, inputs):' if model_dict['inputs'] else '(t, y, params):'
    func_def   = '@jit(nopython=True)\ndef '    if use_numba            else 'def ' 
    func_def   = func_def + func_name + func_args
    states     = vars2code('y', model_dict['states'])
    params     = vars2code('params', model_dict['params'])
    equations  = equations2code(model_dict['equations'], model_dict['states'], indent=1)
    
    if model_dict['inputs']:
        inputs     = vars2code('inputs', model_dict['inputs'])
        result     = '\n\n'.join([result, func_def, states, params, inputs, equations])
    else:
        result     = '\n\n'.join([result, func_def, states, params, equations])
    
    if filename:
        with open(filename, 'w') as file:
            file.write(result)
    return result

def objectives2code(model_dict, objectives, filename='', include_imports=True):
    
    imports       = 'import numpy  as np\nimport pandas as pd'
    states        = obj_vars2code('table', model_dict['states'])
    params        = obj_vars2code('table', model_dict['params'])
    inputs        = obj_vars2code('table', model_dict['inputs']) if model_dict['inputs'] else ''
    time          = "\tt = table['Time']\n"
    eqns_template = equations2code(model_dict['equations'], indent=1)
    result        = imports if include_imports else ''
    for obj_name, obj_eqns in objectives.items():
        
        func_name  = 'objective_{}_{}'.format(model_dict['name'], obj_name)
        func_args  = '(table):' 
        func_def   = 'def ' + func_name + func_args
        equations  = obj_equations2code(obj_eqns, eqns_template, indent=1)
        
        if model_dict['inputs']:
            result = '\n\n'.join([result, func_def, states, params, inputs, time, equations])
        else:
            result = '\n\n'.join([result, func_def, states, params, time, equations])

    if filename:
        with open(filename, 'w') as file:
            file.write(result)
    return result

###############################################################################
#Supporting Functions
###############################################################################
def sp(longest, state):
    return ' '*(longest - len(state) + 1)

def t(indent):
    return '\t'*indent

###############################################################################
#Code Generation for Objective Functions
###############################################################################
def obj_vars2code(name, lst, indent=1):
    longest = len(max(lst, key=len))
    temp    = ["{}{}{}= {}['{}']".format(t(indent), x, sp(longest, x), name, x) for x in lst]
    result  = '\n'.join(temp)
    
    return result
    
def obj_equations2code(obj_eqns, eqns_template, indent=1):
    result = '\n'.join([t(indent) + line.strip() for line in obj_eqns.split('\n')])
    result = '\n\n'.join([eqns_template, result])
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
    equations_ = re.sub('@short .*', lambda match: parse_shorthand(match.group(0)), equations)
    result = '\n'.join([t(indent) + line for line in equations_.split('\n')])
    
    if states:
        diff         = ['d'+state for state in states]
        return_value = '{}return np.array([{}])'.format(t(indent), ', '.join(diff))
        result       = '\n\n'.join([result, return_value])    
    
    return result
        
###############################################################################
#Shorthand Coding
###############################################################################
def parse_shorthand(string):
        
    string_ = split_top_level(string[7:], delimiter=':')
    values  = [[v.strip() for v in split_top_level(vals)] for vals in string_[1:]]
    zipped  = zip(*values)
    expr    = string_[0]
    
    result = '\n'.join([expr.format(*args) for args in zipped])
    
    return result
    
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

if __name__ == '__main__':
    
    __model__ = {'name'     : 'testmodel', 
                 'states'   : ['x', 's', 'p'], 
                 'params'   : ['ks', 'mu_max', 'synp', 'ys'], 
                 'inputs'   : ['b'], 
                 'equations': '\nmu = mu_max*s/(s+ks)\n\n#Differentials\ndx = mu*x\nds = -dx/ys + b\ndp = synp - p*mu', 
                 'meta'     : {}
                 }
    __model__['equations'] += '\n@short d{} =-k*{}: x1, x2, x3: x1, x2, x3'
    # #Not part of the main tests
    # states    = vars2code(list(__model__['states']))
    # params    = vars2code(list(__model__['params']))
    # inputs    = vars2code(list(__model__['inputs']))
    # equations = equations2code(__model__['equations'], list(__model__['states']), indent=1)
    
    # #code generation
    # code = model2code(__model__)
    # print(code)
    
    # objectives = {'obj_1': 'return t, -dx/ys',
    #               'obj_2': 'return t, mu'
    #               }
    
    # states    = obj_vars2code(list(__model__['states']))
    # params    = obj_vars2code(list(__model__['params']))
    # inputs    = obj_vars2code(list(__model__['inputs']))
    
    # eqns_template = equations2code(__model__['equations'], indent=1)
    # equations     = obj_equations2code(obj_eqns, eqns_template, indent=1)
    
    # code = objectives2code(__model__, objectives)
    
    # #Test model function
    # func, code = model2func(__model__)
    
    # y0     = np.array([1, 1, 1])
    # t0     = 0
    # params = np.array([1, 1, 1, 1])
    # inputs = np.array([0])
    
    # dy = func(t0, y0, params, inputs)
    
    # assert all(dy == [0.5, -0.5, 0.5])
    
    # objectives = {'obj_1': 'return t, -dx/ys',
    #               'obj_2': 'return t, mu'
    #               }
    
    # #Test objective functions
    # funcs, code = objectives2func(__model__, objectives)
    
    # y0     = pd.DataFrame([[1, 1, 1], [1.5, 0.5, 1.5]], columns=__model__['states'])
    # t0     = pd.DataFrame([[0], [1]], columns=['Time'])
    # params = pd.DataFrame([[1, 1, 1, 1], [1, 1, 1, 1]], columns=__model__['params'])
    # inputs = pd.DataFrame([[0], [0]], columns=['b'])
    # table  = pd.concat((t0, y0, params, inputs), axis=1)
    
    # xo1, yo1 = funcs['obj_1'](table)
    # assert all(xo1 == [0, 1])
    # assert all(yo1 == [-0.5, -0.5])
    
    # xo2, yo2 = funcs['obj_2'](table)
    # assert all(xo2 == [0, 1])
    # assert all(np.isclose(yo2, [0.5, 0.33333]))