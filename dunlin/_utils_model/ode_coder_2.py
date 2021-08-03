import re
from pathlib import Path

###############################################################################
#Supporting Imports
###############################################################################
import numpy          as np
import pandas         as pd
import scipy.optimize as opt
from   numba  import njit 
#Do NOT remove these imports. They are needed to run dynamically compiled functions.

###############################################################################
#Non-Standard Imports
###############################################################################
try:
    from  .base_error import DunlinBaseError
    from  .funcs      import code2func
    from  .           import funcs as c2f
except Exception as e:
    if Path.cwd() == Path(__file__).parent:
        from  base_error import DunlinBaseError
        from  funcs      import code2func
        import funcs     as c2f
    else:
        raise e

'''
1. Top Level
func_data = make_ode_data(dun_data)

2. High level (but slightly lower)
templates = make_templates(dun_data)
func_data = {}

for model_key, model_data in dun_data.items():
    func_data.setdefault(model_key, {})

    func_data[model_key]['rhs']    = rhs2func(model_key, templates, dun_data)
    func_data[model_key]['exvs']   = exvs2func(model_key, templates, dun_data)
    func_data[model_key]['events'] = events2func(model_key, templates, dun_data) if model_data.get('events') else None
    func_data[model_key]['modify'] = modify2func(model_key, templates, dun_data) if model_data.get('modify') else None
    
'''

###############################################################################
#Globals
###############################################################################
_args = 't', 'y', 'p'

###############################################################################
#High-Level Protocols
###############################################################################
def make_ode_data(dun_data):
    templates = make_templates(dun_data)
    func_data = {}
    
    for model_key, model_data in dun_data.items():
        func_data.setdefault(model_key, {})
    
        func_data[model_key]['rhs']    = rhs2func(model_key, templates, dun_data)
        func_data[model_key]['exvs']   = exvs2func(model_key, templates, dun_data)
        func_data[model_key]['events'] = events2func(model_key, templates, dun_data) if model_data.get('events') else None
        func_data[model_key]['modify'] = modify2func(model_key, templates, dun_data) if model_data.get('modify') else None
    
    return func_data
    
###############################################################################
#High-Level Function Generators
###############################################################################
def rhs2func(*args, **kwargs):
    temp = rhs2code(*args, **kwargs)
    return temp, code2func(temp)

def exvs2func(*args, **kwargs):
    temp  = exvs2code(*args, **kwargs)
    return temp, code2func(temp)

def events2func(*args, **kwargs):
    temp = events2code(*args, **kwargs)
    return temp, {k: code2func(v) for k, v in temp.items()}
    
def event2func(*args, **kwargs):
    temp = event2code(*args, **kwargs)
    return temp, code2func(temp)

def modify2func(*args, **kwargs):
    temp = modify2code(*args, **kwargs)
    return temp, code2func(temp)

###############################################################################
#High Level Code Generators
###############################################################################
#The functions here produce complete, executable code, not just snippets

def rhs2code(model_key, templates, dun_data, numba=True):
    global _args
    
    template   = templates[model_key]
    model_data = dun_data[model_key]
    states     = model_data['states']
    return_val = ', '.join([f'd_{x}' for x in states])
    return_val = f'\treturn np.array([{return_val}])'
    func_name  = 'model_' + model_key
    sections   = [make_def(func_name, *_args, numba=numba), 
                  template,
                  return_val
                  ]
    code       = '\n'.join(sections)
    
    return func_name, code

def exvs2code(model_key, templates, dun_data):
    global _args
    
    template   = templates[model_key]
    model_data = dun_data[model_key]
    exvs       = model_data.get('exvs')
    
    if not exvs:
        return {}
    
    sections      = [None, template, '\t#EXV', None]
    codes         = {}
    
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
        
        codes[exv_name] = func_name, code
    
    return codes

def events2code(model_key, templates, dun_data):
    global _args
    
    event_data = dun_data[model_key].get('events')
    
    if event_data is None:
        return {}
    
    codes = {event_name: event2code(event_name, model_key, templates, dun_data) for event_name in event_data}
            
    return codes

def event2code(event_name, model_key, templates, dun_data):
    event_args = dun_data[model_key]['events'][event_name]
    
    if hasattr(event_args, 'items'):
        trigger    = event_args['trigger']
        assignment = event_args['assignment']
    else:
        trigger = event_args[0]
        assignment = event_args[1]
    
    trigger_    = trigger2code(event_name, trigger, model_key, templates, dun_data)
    assignment_ = assignment2code(event_name, assignment, model_key, templates, dun_data)
    
    return {'trigger': trigger_, 'assignment': assignment_}
    
def trigger2code(event_name, trigger, model_key, templates, dun_data):
    global _args
    
    template   = templates[model_key]
    model_data = dun_data[model_key] 
    func_name  = f'trigger_{model_key}_{event_name}'
    func_def   = make_def(func_name, *_args, numba=False)
    trigger_   = parse_trigger(trigger, **model_data)
    return_val = f'\treturn {trigger_}'
    code       = '\n'.join([func_def, template, '\t#Trigger', return_val])
    
    return func_name, code

def assignment2code(event_name, assignment, model_key, templates, dun_data):
    global _args
    
    template   = templates[model_key]
    model_data = dun_data[model_key]
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
    
def modify2code(model_key, templates, dun_data):
    global _args

    template   = templates[model_key]
    model_data = dun_data[model_key]
    modify     = model_data.get('modify')
    
    if not modify:
        return None, None
    
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
    return '\t#States\n' + lst2code('y', states, indent)

def params2code(params, indent=1):
    return '\t#Params\n' + lst2code('p', params, indent)

def lst2code(name, lst, indent=1):
    if not lst:
        return ''
    
    longest = len(max(lst, key=len))
    temp    = ["{}{}{}= {}[{}]".format(T(indent), x, sp(longest, x), name, i) for i, x in enumerate(lst)]
    result  = '\n'.join(temp) + '\n'
    
    return result

###############################################################################
#Reactions
###############################################################################
def rxns2code(dun_data, model_key, indent=1):
    #Extract from dun_data
    states   = dun_data[model_key]['states']
    rxns     = dun_data[model_key].get('rxns', {})
    rts      = dun_data[model_key].get('rts', {})
    
    #Initialize containers
    rxn_defs = f'{T(indent)}#Reactions\n'
    d_states = {x: f'{T(indent)}d_{x} =' for x in states}
    e_states = [pair for pair in enumerate(states) if pair[1] not in rts] if rts else list(enumerate(states))
    
    if rxns:
        for rxn_num, (rxn_name, rxn_args) in enumerate(rxns.items()):
            
            if hasattr(rxn_args, 'items'):
                rxn_type, stoich, rate  = parse_rxn(dun_data, **rxn_args)
                
            elif hasattr(rxn_args, '__iter__'):
                rxn_type, stoich, rate  = parse_rxn(dun_data, *rxn_args)
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
            if type(expr) != str:
                raise TypeError('Rate must be a string.')
            
            #In the future, a dedicated function will be required 
            #to parse delays
            d_states[state] += f' {expr}'
            
    d_states = f'{T(indent)}#Differentials\n' + '\n'.join(d_states.values())
    temp     = [rxn_defs, d_states, '']
        
    return '\n'.join(temp)

def parse_rxn(dun_data, rxn=None, fwd=None, rev=None, submodel=None, substates=None, subparams=None, _prefix='model_'):
    if rxn is None:
        return _parse_submodel(dun_data, submodel, substates, subparams, _prefix)
    
    elif 'submodel' in rxn:
        try:
            _, submodel_name = rxn.split('submodel ', 1)
        except:
            raise SyntaxError('Improper use of submodel keyword. Make sure the "submodel" keyword and the submodel name are separated.')
        submodel_name = submodel_name.strip()
        
        substates = fwd if substates is None else substates
        subparams = rev if subparams is None else subparams
        
        return _parse_submodel(dun_data, submodel_name, substates, subparams, _prefix)
    
    else:
        stoich, rate = _parse_rxn(rxn, fwd, rev)
        return  'rxn', stoich, rate

def _parse_submodel(dun_data, submodel_key, substates, subparams, _prefix='model_'):
    submodel        = dun_data[submodel_key]
    submodel_states = submodel['states']
    submodel_params = submodel['params']
    
    if hasattr(substates, 'items'):
        y = ', '.join([substates[xsub] for xsub in submodel_states])
        stoich = {x: i for i, x in enumerate(substates.values())}
    else:
        y      =  ', '.join(substates)
        stoich = {x: i for i, x in enumerate(substates)}
        
    if hasattr(subparams, 'items'):
        p = ', '.join([subparams[xsub] for xsub in submodel_params])
    else:
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
class DunlinCodeGenerationError(DunlinBaseError):
    @classmethod
    def invalid(cls, item, description=''):
        return cls.raise_template(f'Invalid {item} definition. {description}', 0)
    
    @classmethod
    def stoichiometry(cls, rxn):
        return cls.raise_template(f'Negative stoichiometry in the reaction: {rxn}', 1)
    
    @classmethod
    def no_numba(cls, msg=''):
        return cls.raise_template('Invalid use of numba.', 2)
    
if __name__ == '__main__':
    import dun_file_reader as dfr

    dun_data0 = dfr.read_file('dun_test_files/M20.dun')
    dun_data1 = dfr.read_file('dun_test_files/M21.dun')
    
    ###############################################################################
    #Part 1: Low Level Code Generation
    ###############################################################################
    funcs  = dun_data0['M1']['funcs']
    vrbs   = dun_data0['M1']['vrbs']
    rxns   = dun_data0['M1']['rxns']
    states = dun_data0['M1']['states']
    
    #Test func def
    name, args = 'test_func', ['a', 'b']
    
    code      = make_def(name, *args)
    test_func = f'{code}\n\treturn [a, b]'
    exec(test_func)
    a, b = 1, 2
    assert test_func(a, b) == [1, 2]
    
    #Test code generation for local functions
    code      = funcs2code(funcs)
    test_func = f'def test_func(v, x, k):\n{code}\n\treturn MM(v, x, k)'
    exec(test_func)
    assert test_func(2, 4, 6) == 0.8
    
    #Test local variable
    code      = vrbs2code(vrbs)
    test_func = f'def test_func(x2, k1):\n{code}\n\treturn sat2'
    exec(test_func)
    assert test_func(1, 1) == 0.5
    
    #Parse single reaction
    stripper = lambda *s: ''.join(s).replace(' ', '').strip()
    r = _parse_rxn(*rxns['r0'])
    
    assert {'x0': '-1', 'x1': '-2', 'x2': '+1'} == r[0]
    assert stripper(rxns['r0'][1], '-', rxns['r0'][2]) == stripper(r[1])
    
    r = _parse_rxn(*rxns['r1'])
    
    assert {'x2': '-1', 'x3': '+1'} == r[0]
    assert stripper(rxns['r1'][1]) == stripper(r[1])
    
    r = _parse_rxn(*rxns['r2'])
    
    assert {'x3': '-1'} == r[0]
    assert stripper(rxns['r2'][1]) == stripper(r[1])
    
    #Test code generation for multiple reactions
    code = rxns2code(dun_data0, 'M1')
    
    MM        = lambda v, x, k: 0
    sat2      = 0.5
    test_func = f'def test_func(x0, x1, x2, x3, x4, p0, p1, p2, p3, p4):\n{code}\treturn [d_x0, d_x1, d_x2, d_x3, d_x4]'
    exec(test_func)
    r = test_func(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    assert r == [-1.0, -2.0, 0.5, -0.5, 1]
    
    #Test code generation for hierarchical models
    #We need to create the "submodel"
    MM         = lambda v, x, k: 0
    code       = rxns2code(dun_data1, 'M2')
    test_func0 = 'def model_M2(*args): return np.array([1, 1])'
    exec(test_func0)
    
    code = rxns2code(dun_data1, 'M3')
    test_func = f'def test_func(t, x0, x1, x2, x3, p0, p1, p2, p3, k2):\n{code}\treturn [d_x0, d_x1, d_x2, d_x3]'
    exec(test_func)
    r = test_func(0, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    assert r == [-1, 2, 1, 1]
    
    dun_data1['M3']['rxns']['r1'] = {'submodel': 'M2', 
                                      'substates': {'xx0': 'x1', 'xx1': 'x2'}, 
                                      'subparams': {'pp0' : 'p0', 'pp1' : 'p1', 'kk1': 'k2'}
                                      }
    
    code = rxns2code(dun_data1, 'M3')
    test_func = f'def test_func(t, x0, x1, x2, x3, p0, p1, p2, p3, k2):\n{code}\treturn [d_x0, d_x1, d_x2, d_x3]'
    exec(test_func)
    r = test_func(0, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    assert r == [-1, 2, 1, 1]
    
    ###############################################################################
    #Part 2: High Level Code Generation
    ###############################################################################
    templates0 = make_templates(dun_data0)
    templates1 = make_templates(dun_data1)
    
    params  = dun_data0['M1']['params']
    exvs    = dun_data0['M1']['exvs']
    events  = dun_data0['M1']['events'] 
    modify  = dun_data0['M1']['modify'] 
    
    #Generate code for ode rhs
    code      = rhs2code('M1', templates0, dun_data0)[1]
    test_func = code.replace('model_M1', 'test_func')

    exec(test_func)
    t  = 0 
    y  = np.ones(5)
    p  = pd.DataFrame(params).values[0]
    dy = test_func(t, y, p)
    assert all( dy == np.array([-0.5, -1,  0,  -1.5 , 2]) )
    
    #Generate code for exv    
    codes     = exvs2code('M1', templates0, dun_data0)
    test_func = codes['r0'][1].replace('exv_M1_r0', 'test_func')

    exec(test_func)
    t  = np.array([0, 1])
    y  = np.ones((5, 2))
    p  = pd.DataFrame(params).values[0]
    r  = test_func(t, y, p)
    assert all(r == 0.5)
    
    #Generate code for single event trigger
    trigger = events['e0'][0] 
    
    code      = trigger2code('e0', trigger, 'M1', templates0, dun_data0)[1]
    test_func = code.replace('trigger_M1_e0', 'test_func')
    exec(test_func)
    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = test_func(t, y, p)
    assert r == 0.5
    
    #Generate code for single event assignment
    assignment = events['e0'][1] 
    
    code      = assignment2code('e0', assignment, 'M1', templates0, dun_data0)[1]
    test_func = code.replace('assignment_M1_e0', 'test_func')
    exec(test_func)
    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = test_func(t, y, p)
    assert r[0][0]              == 5
    assert r[1][0]              == 0.5
    
    #Generate code for single event
    codes = event2code('e0', 'M1', templates0, dun_data0)
    
    test_func = codes['trigger'][1].replace('trigger_M1_e0', 'test_func')
    exec(test_func)
    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = test_func(t, y, p)
    assert r == 0.5
    
    test_func = codes['assignment'][1].replace('assignment_M1_e0', 'test_func')
    exec(test_func)
    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = test_func(t, y, p)
    assert r[0][0]              == 5
    assert r[1][0]              == 0.5
    
    #Generate code for all events
    codes = events2code('M1', templates0, dun_data0)
    
    test_func = codes['e0']['trigger'][1].replace('trigger_M1_e0', 'test_func')
    exec(test_func)
    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = test_func(t, y, p)
    assert r == 0.5
    
    test_func = codes['e0']['assignment'][1].replace('assignment_M1_e0', 'test_func')
    exec(test_func)
    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = test_func(t, y, p)
    assert r[0][0]              == 5
    assert r[1][0]              == 0.5
    
    #Generate modify 
    code      = modify2code('M1', templates0, dun_data0)[1]
    test_func = code.replace('modify_M1', 'test_func')
    exec(test_func)
    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = test_func(y, p, scenario=1)
    assert all( r[0] == np.array([10, 1, 1, 1, 1]) )
    assert all( r[1] == p)
    
    ###############################################################################
    #Part 3A: Function Generation
    ###############################################################################
    #Generate single function from code
    code      = 'x = lambda t: t+1'
    scope     = {}
    test_func = code2func(['x', code])
    assert test_func(5) == 6
    
    #Generate multiple functions from codes
    #The second function requires access to the first one
    codes     = {'fx': ['x', 'def x(t):\n\treturn t+1'], 
                 'fy': ['y', 'def y(t):\n\treturn x(t)+2']
                 }
    r         = code2func(codes)
    test_func = r['fx']
    assert test_func(5) == 6
    test_func = r['fy']
    assert test_func(5) == 8
    
    ###############################################################################
    #Part 3B: Function Generation
    ###############################################################################
    templates0 = make_templates(dun_data0)
    templates1 = make_templates(dun_data1)
    
    params  = dun_data0['M1']['params']
    exvs    = dun_data0['M1']['exvs']
    events  = dun_data0['M1']['events'] 
    modify  = dun_data0['M1']['modify'] 
    
    #Generate rhs function
    code, func = rhs2func('M1', templates0, dun_data0)
    t  = 0 
    y  = np.ones(5)
    p  = pd.DataFrame(params).values[0]
    dy = func(t, y, p)
    assert all( dy == np.array([-0.5, -1,  0,  -1.5 , 2]) )
    
    #Generate exv functions
    codes, funcs = exvs2func('M1', templates0, dun_data0)
    code, func   = codes['r0'], funcs['r0']
    
    t  = np.array([0, 1])
    y  = np.ones((5, 2))
    p  = pd.DataFrame(params).values[0]
    r  = func(t, y, p)
    assert all(r == 0.5)
    
    #Generate event functions for one event
    codes, funcs = event2func('e0', 'M1', templates0, dun_data0)
    
    func = funcs['trigger']

    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = func(t, y, p)
    assert r == 0.5
    
    func = funcs['assignment']

    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = func(t, y, p)
    assert r[0][0]              == 5
    assert r[1][0]              == 0.5
    
    #Generate event functions for all events
    codes, funcs = events2func('M1', templates0, dun_data0)
    
    func = funcs['e0']['trigger']

    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = func(t, y, p)
    assert r == 0.5
    
    func = funcs['e0']['assignment']

    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = func(t, y, p)
    assert r[0][0]              == 5
    assert r[1][0]              == 0.5
    
    #Generate modify 
    code, func = modify2func('M1', templates0, dun_data0)
    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = func(y, p, 1)
    assert all( r[0] == np.array([10, 1, 1, 1, 1]) )
    assert all( r[1] == p)
    
    ###############################################################################
    #Part 4: Top Level Functions
    ###############################################################################
    #Create functions from dun_data
    func_data = make_ode_data(dun_data0)
    
    #Generate rhs function
    code, func = func_data['M1']['rhs']
    t  = 0 
    y  = np.ones(5)
    p  = pd.DataFrame(params).values[0]
    dy = func(t, y, p)
    assert all( dy == np.array([-0.5, -1,  0,  -1.5 , 2]) )
    
    #Generate exv functions
    codes, funcs = func_data['M1']['exvs']
    code, func   = codes['r0'], funcs['r0']
    
    t  = np.array([0, 1])
    y  = np.ones((5, 2))
    p  = pd.DataFrame(params).values[0]
    r  = func(t, y, p)
    assert all(r == 0.5)
    
    #Generate event functions for all events
    codes, funcs = func_data['M1']['events']
    
    func = funcs['e0']['trigger']

    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = func(t, y, p)
    assert r == 0.5
    
    func = funcs['e0']['assignment']

    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = func(t, y, p)
    assert r[0][0]              == 5
    assert r[1][0]              == 0.5
    
    #Generate modify 
    code, func = func_data['M1']['modify']
    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = func(y, p, 1)
    assert all( r[0] == np.array([10, 1, 1, 1, 1]) )
    assert all( r[1] == p)
    