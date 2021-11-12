import re
from . import dun_string_reader as dsr

subsection_types = {'tspan'             : ['dun', 'tspan'            ],
                    'states'            : ['dun', 'states'           ],
                    'params'            : ['dun', 'params'           ],
                    'rxns'              : ['dun', 'rxns'             ],
                    'reactions'         : ['dun', 'rxns'             ],
                    'rts'               : ['dun', 'rts'              ],
                    'rates'             : ['dun', 'rts'              ],
                    'vrbs'              : ['dun', 'vrbs'             ],
                    'variables'         : ['dun', 'vrbs'             ],
                    'funcs'             : ['dun', 'funcs'            ],
                    'functions'         : ['dun', 'funcs'            ],
                    'evs'               : ['dun', 'events'           ],
                    'events'            : ['dun', 'events'           ],
                    'comps'             : ['dun', 'comps'            ],
                    'compartments'      : ['dun', 'comps'            ],
                    'exvs'              : ['py',  'exvs'             ],
                    'modify'            : ['py',  'modify'           ],
                    'int_args'          : ['dun', 'int_args'         ],
                    'sim_args'          : ['dun', 'sim_args'         ],
                    'optim_args'        : ['dun', 'optim_args'       ],
                    'strike_goldd_args' : ['dun', 'strike_goldd_args'],
                    'sg_args'           : ['dun', 'strike_goldd_args']
                    }    

def parse_element(element, subsection_type):
    t, arg_name = subsection_types.get(subsection_type)
    
    if t == 'dun':
        substituted = substitute_dun(element)
        result      = {} 
        
        for s in substituted:
            data   = dsr.read_dun(s)
            result = {**result, **data} 

        return result
    else:
        name, formatted = dsr.format_indent(element)
        substituted     = substitute_py(formatted)
        
        return {name: substituted}

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
    template, h_vals, v_vals = split_dun_element(element)
    
    if len(h_vals) > 1:
        it       = iter(h_vals.values())
        n_values = len(next(it))
        if not all(len(l) == n_values for l in it):
            raise DunUnequalShorthandError(element)
    
    if len(v_vals) > 1:
        it       = iter(v_vals.values())
        n_values = len(next(it))
        if not all(len(l) == n_values for l in it):
            raise DunUnequalShorthandError(element)
    
    template_ = substitute_horizontal(template, h_vals)
    strings   = substitute_vertical(template_, v_vals)
        
    return strings

def split_dun_element(element):
    def update(chunk, dct):
        try:
            field, vals = chunk.split(',', 1)
            field       = field.strip()
            vals        = split_top_delimiter(vals)
            dct[field]  = vals
            return 
        except:
            raise DunSplitShorthandError(repr(chunk))
    
    temp     = re.findall('([^$?]*)([$?])([^$?]*)', element, flags=re.S)
    template = ''
    h_vals   = {}
    v_vals   = {}

    if temp:
        template = ''
        
        for i, marker, chunk in temp:
            i = i.strip()
            if template and i:
                raise DunWrongShorthandError(element)
            elif not template:
                template = i
            
            if marker == '$':
                update(chunk, v_vals)
            else:
                update(chunk, h_vals)
            
    else:
        template = element
    
    return template, h_vals, v_vals

def split_top_delimiter(string, delimiter=','):
    string = string.strip()
    i0     = 0
    depth  = 0
    chunks = []
    for i, char in enumerate(string):
        if char == delimiter and depth == 0:
            
            chunk = string[i0: i].strip()
            chunks.append(chunk)

            i0    = i + 1
        elif char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1

    if i0 < len(string) - 1:
        chunk = string[i0: ].strip()
        chunks.append(chunk)

    return chunks
    
def substitute_horizontal(template, h_vals):
    
    keys = h_vals.keys()
    vals = h_vals.values()
    vals = list(zip(*vals))
    
    def repl(m):
        try:
            result = [m[1].format(**dict(zip(keys, v))) for v in vals]
        except KeyError:
            raise DunWrongShorthandError(template)
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
        raise DunWrongShorthandError(template)
    return strings

class DunSplitShorthandError(Exception):
    def __init__(self, element):
        return super().__init__(f'Could not split the following: {element}')
    
class DunWrongShorthandError(Exception):
    def __init__(self, template):
        return super().__init__(f'Wrong shorthand keys/values: {template}')
    
class DunUnequalShorthandError(Exception):
    def __init__(self, element):
        return super().__init__(f'The element has shorthands with values of unequal length: {element}')
  
