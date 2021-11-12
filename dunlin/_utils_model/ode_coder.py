import textwrap as tw
import re

###############################################################################
#Supporting Imports
###############################################################################
import numpy          as np
import pandas         as pd

###############################################################################
#Non-Standard Imports
###############################################################################
from  .funcs      import code2func
'''
Top Level function is make_ode_data
func_data = make_ode_data(dun_data)
'''

###############################################################################
#Globals
###############################################################################
_args = 't', 'states', 'params'

###############################################################################
#High-Level Protocols
###############################################################################
def make_ode_data(model_data, numba=True):
    template  = make_template(model_data)
    func_data = {}
    
    func_data['rhs'   ] = rhs2func(template, model_data, numba=numba)
    func_data['sim'   ] = sim2func(template, model_data)
    func_data['exvs'  ] = exvs2func(template, model_data) 
    func_data['events'] = events2func(template, model_data) if model_data.get('events') else []
    func_data['modify'] = modify2func(template, model_data) if model_data.get('modify') else None
    func_data['eqns']   = template
    func_data['exvrhs'] = exvrhs2func(model_data, func_data['rhs'].code)
    
    return func_data
    
###############################################################################
#High-Level Function Generators
###############################################################################
def rhs2func(*args, **kwargs):
    temp = rhs2code(*args, **kwargs)
    return code2func(temp)

def exvrhs2func(model_data, rhscode):
    temp = rhs2exvrhs(model_data, rhscode)
    return code2func(temp)

def sim2func(*args, **kwargs):
    temp = sim2code(*args, **kwargs)
    return code2func(temp)

def exvs2func(*args, **kwargs):
    temp  = exvs2code(*args, **kwargs)
    return code2func(temp)

def events2func(*args, **kwargs):
    temp = events2code(*args, **kwargs)
    return {k: code2func(v) for k, v in temp.items()}
    
def event2func(*args, **kwargs):
    temp = event2code(*args, **kwargs)
    return code2func(temp)

def modify2func(*args, **kwargs):
    temp = modify2code(*args, **kwargs)
    return code2func(temp)

###############################################################################
#High Level Code Generators
###############################################################################
#The functions here produce complete, executable code.
#They do NOT check if model_data has the relevant key/value
def rhs2code(template, model_data, numba=True):
    global _args
    
    model_key  = model_data['model_key']
    states     = model_data['states']
    return_val = ', '.join([f'd_{x}' for x in states])
    return_val = f'\treturn np.array(({return_val}))'
    
    func_name  = 'model_' + model_key
    sections   = [make_def(func_name, *_args, numba=numba), 
                  template,
                  return_val
                  ]
    code       = '\n'.join(sections)
    
    return func_name, code

def rhs2exvrhs(model_data, rhscode):
    model_key  = model_data['model_key']
    func_name  = 'modelexv_' + model_key
    code       = rhscode.replace('\treturn np.array', '\treturn np.stack')
    code       = code.replace('model_', 'modelexv_')
    
    return func_name, code

def sim2code(template, model_data, numba=True):
    global _args
    
    #Create inner function that is numba compatible
    model_key  = model_data['model_key']
    states     = model_data['states']
    vrbs       = model_data.get('vrbs', [])
    vrbs       = [] if vrbs is None else vrbs
    diffs      = [f'd_{x}' for x in states]
    all_vars   = list(states) + list(vrbs) + diffs + [_args[0]]
    return_val = ', '.join(all_vars) 
    return_val = f'\treturn {return_val}'
    func_name  = 'helper' 
    sections   = [make_def(func_name, *_args, numba=numba), 
                  template,
                  return_val
                  ]
    inner_code = '\n'.join(sections)
    
    #Indent the code
    func_name  = 'sim_' + model_key
    inner_code = tw.indent(inner_code, '\t')
    keys       = f'\tkeys = {all_vars}'
    return_val = f'\treturn dict(zip(keys, helper({", ".join(_args)})))'
    sections   = [make_def(func_name, *_args, numba=False), 
                  keys,
                  inner_code,
                  return_val
                  ]
    code       = '\n'.join(sections)
    
    return func_name, code

def exvs2code(template, model_data):
    global _args
    model_key = model_data['model_key']
    exvs      = model_data['exvs'     ]
    sections  = [None, template, '\t#EXV', None]
    codes     = {}

    #Create default exv func
    vrbs     = list(model_data['vrbs']) if model_data.get('vrbs') else []
    rxns     = list(model_data['rxns']) if model_data.get('rxns') else []
    diffs    = [f'd_{x}' for x in model_data['states']]
    combined = vrbs + rxns + diffs
    exv_code = '\treturn {' + ', '.join([f'{repr(i)}: {i}' for i in combined]) + '}'
    exvs     = {**{'all__': exv_code}, **exvs} if exvs else {'all__': exv_code}
    
    for exv_name, exv_code in exvs.items():
        if '@numba' in exv_code:
            exv_code_ = exv_code.replace('@numba', '')
            numba     = True
        else:
            exv_code_ = exv_code
            numba     = False
            
        #Substitute into sections
        func_name    = 'exv_' + model_key + '_' + exv_name
        sections[0]  = make_def(func_name, *_args, numba=numba)
        sections[-1] = exv_code_
        code         = '\n'.join(sections)
        code         = code.replace('model_', 'modelexv_')
        codes[exv_name] = func_name, code
    
    return codes

def events2code(template, model_data):
    global _args
    codes = {event_name: event2code(event_name, template, model_data) for event_name in model_data['events']}
            
    return codes

def event2code(event_name, template, model_data):
    event_args = model_data['events'][event_name]
    
    if hasattr(event_args, 'items'):
        trigger    = event_args['trigger']
        assignment = event_args['assignment']
    else:
        trigger = event_args[0]
        assignment = event_args[1]
    
    trigger_    = trigger2code(event_name, trigger,template, model_data)
    assignment_ = assignment2code(event_name, assignment, template, model_data)
    
    return {'trigger': trigger_, 'assignment': assignment_}
    
def trigger2code(event_name, trigger, template, model_data):
    global _args
    
    model_key  = model_data['model_key']
    func_name  = f'trigger_{model_key}_{event_name}'
    func_def   = make_def(func_name, *_args, numba=False)
    trigger_   = parse_trigger(trigger, **model_data)
    return_val = f'\treturn {trigger_}'
    code       = '\n'.join([func_def, template, '\t#Trigger', return_val])
    
    return func_name, code

def assignment2code(event_name, assignment, template, model_data):
    global _args
    
    model_key  = model_data['model_key']
    states     = model_data['states']
    params     = model_data.get('params')
    
    if type(assignment) == str:
        assignment_  = parse_assignment(assignment, **model_data)
        assignment_  = f'\t#Assignment\n\t{assignment_}'
    elif hasattr(assignment, '__iter__'):
        assignment_ = [parse_assignment(a, **model_data) for a in assignment]
        assignment_ = '\n\t'.join(assignment_)
        assignment_ = f'\t#Assignment\n\t{assignment_}' 
    else:
        raise TypeError('Error in generating code for event assignment. Assignments must be list-like or a string: {assignment}')
        
    template_    = template.split('#Equations')[0]
    func_name    = f'assignment_{model_key}_{event_name}'
    func_def     = make_def(func_name, *_args, numba=False)
    new_y        = f'\n\tnew_y = np.array([{", ".join([state for state in states])}])'
    new_p        = f'\n\tnew_p = np.array([{", ".join([param for param in params])}])'
    return_val   = f'{new_y}{new_p}\n\treturn new_y, new_p'
    code         = '\n'.join([func_def, template_, assignment_, return_val])
    
    return func_name, code
    
def modify2code(template, model_data):
    global _args
    
    model_key = model_data['model_key']
    states    = model_data['states']
    params    = model_data['params']
    modify    = model_data.get('modify')
    
    if len(modify) > 1:
        raise DunlinCodeGenerationError('modify', 'Only one modify function is allowed.')
    
    modify = '\n'.join(modify.values())
    
    if '@numba' in modify:
        raise DunlinCodeGenerationError.no_numba('Numba cannot be used for modifier function.')
    
    template_  = template.split('#Equations')[0]
    modify_    = f'\t#Modify\n{modify}'
    func_name  = f'modify_{model_key}'
    func_def   = make_def(func_name, *_args[1:], 'scenario', numba=False)
    new_y      = f'\n\tnew_y = np.array([{", ".join([x for x in states])}])'
    new_p      = f'\n\tnew_p = np.array([{", ".join([p for p in params])}])'
    return_val = f'{new_y}{new_p}\n\treturn new_y, new_p'
    code       = '\n'.join([func_def, template_, modify_, return_val])
    
    return func_name, code

def make_templates(dun_data):
    templates = {}
    for model_key, data in dun_data.items():
        states = data['states']
        params = data['params']
        funcs  = data.get('funcs')
        vrbs   = data.get('vrbs')
        
        rcode    = rxns2code(dun_data, model_key)
        sections = [states2code(states), 
                    params2code(params), 
                    '\t#Equations',
                    funcs2code(funcs),
                    vrbs2code(vrbs), 
                    rcode
                    ]
        
        templates[model_key] = '\n'.join([s for s in sections if s])
    return templates

def make_template(model_data):
    states = model_data['states']
    params = model_data['params']
    funcs  = model_data.get('funcs')
    vrbs   = model_data.get('vrbs')
    
    rcode    = rxns2code(model_data)
    sections = [states2code(states), 
                params2code(params), 
                '\t#Equations',
                funcs2code(funcs),
                vrbs2code(vrbs), 
                rcode
                ]
    
    template = '\n'.join([s for s in sections if s])
    return template

###############################################################################
#Function Definition
###############################################################################
def make_def(func_name, *args, numba=True):
    args_ = ', '.join(args)
    return f'@njit\ndef {func_name}({args_}):' if numba else f'def {func_name}({args_}):'

###############################################################################
#States and Params
###############################################################################
def states2code(states, indent=1):
    global _args
    return '\t#States\n' + lst2code(_args[1], states, indent)

def params2code(params, indent=1):
    global _args
    return '\t#Params\n' + lst2code(_args[2], params, indent)

def lst2code(name, lst, indent=1):
    if type(lst) == pd.DataFrame:
        return lst2code(name, list(lst.columns), indent)
    elif type(lst) == pd.Series:
        return lst2code(name, list(lst.index), indent)
    elif not len(lst):
        return ''
    
    longest = len(max(lst, key=len))
    temp    = ["{}{}{}= {}[{}]".format(T(indent), x, sp(longest, x), name, i) for i, x in enumerate(lst)]
    result  = '\n'.join(temp) + '\n'
    
    return result

###############################################################################
#Reactions
###############################################################################
def rxns2code(model_data, indent=1):
    #Extract from model_data
    states   = model_data['states']
    rxns     = model_data.get('rxns', {})
    rts      = model_data.get('rts', {})
    
    #Initialize containers
    rxn_defs = f'{T(indent)}#Reactions\n'
    d_states = {x: f'{T(indent)}d_{x} =' for x in states}
    e_states = [pair for pair in enumerate(states) if pair[1] not in rts] if rts else list(enumerate(states))
    
    if rxns:
        for rxn_num, (rxn_name, rxn_args) in enumerate(rxns.items()):
            
            if hasattr(rxn_args, 'items'):
                rxn_type, stoich, rate  = parse_rxn(model_data, **rxn_args)
                
            elif hasattr(rxn_args, '__iter__'):
                rxn_type, stoich, rate  = parse_rxn(model_data, *rxn_args)
            else:
                raise DunlinCodeGenerationError.invalid('rxns')
            
            if rxn_type == 'rxn':
                for i, x in e_states:
                    n = stoich.get(x)
                    if n:
                        d_states[x] += f' {n}*{rxn_name}'
            else:
                for i, x in e_states:
                    n = stoich.get(x)
                    if n is not None:
                        d_states[x] += f' +{rxn_name}[{n}]'

            rxn_defs += f'{T(indent)}{rxn_name} = {rate}\n' 

    if rts:
        for state, expr in rts.items():
            if type(expr) != str and not isnum(expr):
                raise TypeError('Rate must be a string.')
            
            #In the future, a dedicated function will be required 
            #to parse delays
            d_states[state] += f' {expr}'

    d_states = f'{T(indent)}#Differentials\n' + '\n'.join(d_states.values())
    temp     = [rxn_defs, d_states, '']
        
    return '\n'.join(temp)

def parse_rxn(model_data, rxn=None, fwd=None, rev=None, submodel=None, substates=None, subparams=None, _prefix='model_'):
    if rxn is None:
        return _parse_submodel(model_data, submodel, substates, subparams, _prefix)
    
    elif 'submodel' in rxn:
        try:
            _, submodel_name = rxn.split('submodel ', 1)
        except:
            raise SyntaxError('Improper use of submodel keyword. Make sure the "submodel" keyword and the submodel name are separated.')
        submodel_name = submodel_name.strip()
        
        substates = fwd if substates is None else substates
        subparams = rev if subparams is None else subparams
        
        return _parse_submodel(model_data, submodel_name, substates, subparams, _prefix)
    
    else:
        stoich, rate = _parse_rxn(rxn, fwd, rev)
        return  'rxn', stoich, rate

def _parse_submodel(model_data, submodel_key, substates, subparams, _prefix='model_'):
    if hasattr(substates, 'items') or hasattr(subparams, 'items'):
        raise NotImplementedError('Substates and subparams should be list-like.')
        
    y      =  ', '.join(substates)
    stoich = {x: i for i, x in enumerate(substates)}
    p      =  ', '.join(subparams) 

    y      = f'np.array([{y}])'
    p      = f'np.array([{p}])'

    rate   = _prefix + submodel_key + f'(t, {y}, {p})'

    return 'submodel', stoich, rate

def _parse_rxn(rxn, fwd, rev=None):
    def get_stoich(rct, minus=False):
        rct_ = rct.strip().split('*')

        if len(rct_) == 1:
            n, x = 1, rct_[0].strip()
        else:
            try:
                n, x = rct_[0].strip(), rct_[1].strip()
                    
            except Exception as e:
                raise e
                
            if str2num(n) < 0:
                    raise DunlinCodeGenerationError.stoichiometry(rxn)
                    
        if minus:
            return x, f'-{n}'
        else:
            return x, f'+{n}'

    try:
        rcts, prds = rxn.split('>')
    except:
        raise DunlinCodeGenerationError.invalid('reaction', 'Use a ">" to separate reactants and products.')
    
    rcts   = [get_stoich(rct, minus=True)  for rct in rcts.split('+')] if rcts.strip() else []
    prds   = [get_stoich(prd, minus=False) for prd in prds.split('+')] if prds.strip() else []
    stoich = dict(rcts + prds)
    rate   = fwd.strip() + ' - ' + rev.strip() if rev else fwd.strip()
    
    if not rate:
        raise DunlinCodeGenerationError.invalid('rate')
    
    return stoich, rate
    
def str2num(x):
    try:
        return int(x)
    except:
        pass
    try:
        return float(x)
    except Exception as e:
        raise e  

###############################################################################
#Local Variables
###############################################################################
def vrbs2code(vrbs, indent=1):
    if not vrbs:
        return ''
    
    code = [f'{T(indent)}{vrb} = {expr}\n' for vrb, expr in vrbs.items()]
    code = '\t#Variables\n' + ''.join(code)
    return code

###############################################################################
#Markup Function Substitution
###############################################################################
def funcs2code(funcs):
    if not funcs:
        return ''
    
    code = ''
    for signature, value in funcs.items():      
        if hasattr(value, 'items'):
            code += parse_func(signature, **value) + '\n' 
        elif hasattr(value, 'strip'):
            code += parse_func(signature, value)  + '\n' 
        elif hasattr(value, '__iter__'):
            code += parse_func(signature, *value)  + '\n' 
        else:
            raise TypeError('funcs should be a dictionary, string or list.')
    
    return code

def parse_func(signature, value, indent=1):
    code = '\t#Functions\n' + f'{T(indent)}def {signature}:\n{T(indent+1)}return {value}'
    return code
    
###############################################################################
#Events
###############################################################################
def parse_trigger(string, states, params, rxns=None, vrbs=None, rts=None, funcs=None, **_kwargs):
    pattern = '([^<>=]*)([<>=][=]?)([^<>=]*)'
    temp    = re.findall(pattern, string)
    
    if len(temp) != 1:
        raise DunlinCodeGenerationError.invalid('trigger.')
        
    lhs, op, rhs = temp[0]

    if '<' in op:
        return f'{rhs.strip()} - {lhs.strip()}'
    else:
        return f'{lhs.strip()} - {rhs.strip()}'
    
def parse_assignment(string, states, params, rxns=None, vrbs=None, rts=None, funcs=None, **_kwargs):
    try:
        lhs, rhs = string.split('=')
        lhs      = lhs.strip()
        rhs      = rhs.strip()
    except:
        raise DunlinCodeGenerationError.invalid('assignment')
    
    return lhs + ' = ' + rhs
    
###############################################################################
#Supporting Functions
###############################################################################
#@njit
def T(indent):
    return '\t'*indent

#@njit
def sp(longest, state):
    return ' '*(longest - len(state) + 1)

def isnum(x):
    try:
        float(x)
        return True
    except:
        return False

###############################################################################
#Dunlin Exceptions
###############################################################################
class DunlinCodeGenerationError(Exception):
    @classmethod
    def invalid(cls, item, description=''):
        return cls(f'Invalid {item} definition. {description}')
    
    @classmethod
    def stoichiometry(cls, rxn):
        return cls(f'Negative stoichiometry in the reaction: {rxn}')
    
    @classmethod
    def no_numba(cls, msg=''):
        return cls('Invalid use of numba.')
    
