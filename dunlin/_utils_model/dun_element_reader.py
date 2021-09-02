import numpy  as np
import pandas as pd
import re

###############################################################################
#Non-Standard Imports
###############################################################################

import dunlin._utils_model.dun_string_reader as dsr
from  .base_error  import DunlinBaseError
from  .custom_eval import safe_eval as eval

###############################################################################
#Dunlin Exceptions
###############################################################################
class DunlinShorthandError(DunlinBaseError):
    @classmethod
    def missing(cls, template):
        return cls.raise_template(f'Template has shorthand but missing or wrong keys/values: {template}', 0)
    
    @classmethod
    def len(cls, element):
        return cls.raise_template(f'The element has shorthands with values of unequal length: {element}', 1)
    
    
class DunlinElementError(SyntaxError, DunlinBaseError):
    @classmethod
    def nesting(cls, line):
        return cls.raise_template(f'This line has more than two levels of nesting: {line}', 1)
    
    @classmethod
    def name(cls, name):
        return cls.raise_template(f'Invalid name: {name}', 2)
    
    @classmethod
    def repeat(cls, arg0, arg1=None):
        if arg1:
            return cls.raise_template(f'Repeated definition of {arg0} in {arg1}', 4)
        else:
            return cls.raise_template(f'Repeated definition of {arg0}', 4)
    
    @classmethod
    def shorthand_order(cls, element, e=None):
        global horizontal_delimiter, vertical_delimiter
        return cls.raise_template(f'Shorthand in wrong order. {horizontal_delimiter} comes before {vertical_delimiter}', 5)
     
    @classmethod
    def py(cls, string):
        return cls.raise_template(f'Missing code or invalid subsection format: {string}', 7)
    
    @classmethod
    def value(cls, element):
        return cls.raise_template(f'Invalid value for {element}.')
    
###############################################################################
#Wrapper and Substitutors for Parsers
###############################################################################
def element_type(string='dun', **kwargs):
    def wrapper(func):
        def helper(element):
            if string == 'dun':
                substituted = substitute_dun(element)
                result      = {} 
                
                for s in substituted:
                    data   = dsr.read_dun(s, **kwargs)
                    data   = func(data)
                    result = {**result, **data} 

                return result
            else:
                name, formatted = dsr.format_indent(element)
                substituted     = substitute_py(formatted)
                
                dsr.validate_key(name)
                return {name: func(substituted)}
        return helper
    return wrapper

###############################################################################
#py Shorthand
###############################################################################
def substitute_py(element):
    pattern     = '([ \t]+)@(\w*) (.*(\W*[~!].*)*)'
    substituted = re.sub(pattern, replace_shorthand, element, flags=re.M)
    
    return substituted

def replace_shorthand(m):
    indent = m[1] 
    stype  = m[2]
    if stype == 'numba':
        return m[0]
    
    expr   = m[3].strip()
    link   = '\n' + indent
    result = []
    
    if stype == 'short':
        result = substitute_dun(expr)
        result = [r.strip() for r in result]
        result = link + link.join(result)
    else:
        raise TypeError(f'Unknown shorthand: {stype}')
    return result

###############################################################################
#dun Shorthand
###############################################################################    
def substitute_dun(element):
    try:
        template, h_vals, v_vals = split_dun_element(element)
    except:
        raise DunlinShorthandError.missing(element)
    
    if len(h_vals) > 1:
        it       = iter(h_vals.values())
        n_values = len(next(it))
        if not all(len(l) == n_values for l in it):
            raise DunlinShorthandError.len(element)
    
    if len(v_vals) > 1:
        it       = iter(v_vals.values())
        n_values = len(next(it))
        if not all(len(l) == n_values for l in it):
            raise DunlinShorthandError.len(element)
    
    template_ = substitute_horizontal(template, h_vals)
    strings   = substitute_vertical(template_, v_vals)
    
    for string in strings:
        if '{' in string:
            raise DunlinShorthandError.missing(element)
        
    return strings

def split_dun_element(element):
    def update(chunk, dct):
        field, vals = chunk.split(',', 1)
        field       = field.strip()
        vals        = eval_sub(vals)
        dct[field]  = vals
        return 
    
    template = ''
    v_vals   = {}
    h_vals   = {}
    i0       = 0
    curr     = None 
    for i, char in enumerate(element):
        if char == '!':
            chunk = element[i0:i]
            if template:
                update(chunk, curr)
            else:
                template = chunk
            i0   = i + 1
            curr = h_vals
        elif char == '~':
            chunk = element[i0:i]
            if template:
                update(chunk, curr)
            else:
                template = chunk
            i0   = i + 1
            curr = v_vals
    
    chunk = element[i0:]
    if template:
        update(chunk, curr)
    else:
        template = chunk
    
    return template, h_vals, v_vals
    
def substitute_horizontal(template, h_vals):
    
    keys = h_vals.keys()
    vals = h_vals.values()
    vals = list(zip(*vals))
    
    def repl(m):
        try:
            result = [m[1].format(**dict(zip(keys, v))) for v in vals]
        except KeyError:
            raise DunlinShorthandError.missing(template)
        return ', '.join(result)
         
    string = re.sub('<([^>]*)>', repl, template) 
    
    return string

def substitute_vertical(template, v_vals):
    if not v_vals:
        return [template]
    
    keys = v_vals.keys()
    vals = v_vals.values()
    vals = list(zip(*vals))
    
    try:
        strings = [template.format(**dict(zip(keys, v))) for v in vals]
    except KeyError:
        raise DunlinShorthandError.missing(template)
    return strings

def eval_sub(string):
    string_ = string.rstrip()
    if string_[-1] == ',':
        string_ = string_[:-1]
        
    lst    = dsr.read_dun(string_, expect=list, flat_reader=lambda x:x)
    result = []
    temp   = []
    for i in lst:
        if 'range' in i or 'linspace' in i:
            temp = eval(i, locals={'linspace': np.linspace})
            result += [str(x) for x in temp]     
        else:
            result.append(i.strip())
    return result
        
def check_fields(template):
    chunks      = template.split('{')[1:]
    uses_fields = None
    for chunk in chunks:
        try:
            field, _ = chunk.split('}')
        except:
            raise Exception()
        field = field.strip()

        if uses_fields is None:
            uses_fields = len(field) > 0
        elif not field and uses_fields:
            raise Exception('Inconsistent shorthand')
        elif field and not uses_fields:
            raise Exception('Inconsistent shorthand')
    return uses_fields

###############################################################################
#IMPORTANT STUFF
###############################################################################
'''
Place the parsers for individual arguments here. The rules follow:
    1. Decorate with element_type('dun') or element_type('py')
        1. This wraps the substitutions and parsing.
    2. Name the function parse_<argument> using the full name of the argument
        1. This is doesn't affect performance but just follow it TYVM
    3. Set the signature to data. 
        1. The input will be a dictionary {subsection_name: parsed_data}
    4. Add the relevant modifications.
        1. Do NOT substitution and parse. This was already done using the decorator.
        2. Return data if no modifications are required. 
    5. Update the dictionary called parsers.
        1. The key is the subsection name
        2. The value is a list of the form [parser, argument name]
        3. Decoupling the subsection and argument names allows for aliases.
        4. However, aliases should only be used for core arguments.
'''

##############################################################################
#DN Type Parsers
###############################################################################
@element_type('dun', min_depth=1, max_depth=1)
def parse_tspan(data):
    if not data:
        return {}
 
    for key in data.keys():
        tspan  = data[key]
        result = []
        nums   = []
        for i in tspan:
            if type(i) == str:
                if 'linspace' in i:
                    values = eval(i, locals={'linspace': np.linspace})
                    result.append(values)
                elif 'logspace' in i:
                    values = eval(i, locals={'logspace': np.logspace})
                    result.append(values)
                elif 'range' in i:
                    values = eval(f'{i.strip()}')
                    result.append(values)
                else:
                    raise DunlinElementError.value('tspan')
            else:
                nums.append(i)
                
        result    = np.concatenate(result+[nums])
        result    = np.unique(result)
        data[key] = result
            
    return data

@element_type('dun', min_depth=1, max_depth=1)
def parse_states(data):
    return data

@element_type('dun', min_depth=1, max_depth=1)
def parse_params(data):
    return data
    
@element_type('dun', min_depth=1, max_depth=3)
def parse_reactions(data):
    return data

@element_type('dun', min_depth=0, max_depth=1)
def parse_rates(data):
    return data

@element_type('dun', min_depth=0, max_depth=0)
def parse_variables(data):
    return data

@element_type('dun', min_depth=0, max_depth=0)
def parse_functions(data):
    return data

@element_type('dun', min_depth=1, max_depth=2)
def parse_events(data):
    return data

@element_type('dun', min_depth=1, max_depth=2)
def parse_compartments(data):
    return data

@element_type('dun')
def parse_int_args(data):
    return data

@element_type('dun')
def parse_sim_args(data):
    return data

@element_type('dun')
def parse_optim_args(data):        
    return data

@element_type('dun')
def parse_strike_goldd_args(data):
    return data

###############################################################################
#PY Type Parsers
###############################################################################
@element_type('py')
def parse_exvs(data):
    return data

@element_type('py')
def parse_modify(data):
    return data

###############################################################################
#Index of Parsers
###############################################################################
#subsection name: [function, argument name for instantiating model]
parsers = {'tspan'             : [parse_tspan,             'tspan'            ],
           'states'            : [parse_states,            'states'           ],
           'params'            : [parse_params,            'params'           ],
           'rxns'              : [parse_reactions,         'rxns'             ],
           'reactions'         : [parse_reactions,         'rxns'             ],
           'rts'               : [parse_rates,             'rts'              ],
           'rates'             : [parse_rates,             'rts'              ],
           'vrbs'              : [parse_variables,         'vrbs'             ],
           'variables'         : [parse_variables,         'vrbs'             ],
           'funcs'             : [parse_functions,         'funcs'            ],
           'functions'         : [parse_functions,         'funcs'            ],
           'evs'               : [parse_events,            'events'           ],
           'events'            : [parse_events,            'events'           ],
           'comps'             : [parse_compartments,      'comps'            ],
           'compartments'      : [parse_compartments,      'comps'            ],
           'exvs'              : [parse_exvs,              'exvs'             ],
           'modify'            : [parse_modify,            'modify'           ],
           'int_args'          : [parse_int_args,          'int_args'         ],
           'sim_args'          : [parse_sim_args,          'sim_args'         ],
           'optim_args'        : [parse_optim_args,        'optim_args'       ],
           'strike_goldd_args' : [parse_strike_goldd_args, 'strike_goldd_args'],
           'sg_args'           : [parse_strike_goldd_args, 'strike_goldd_args']
           }    


