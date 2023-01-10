import numpy as np
import textwrap as tw
from collections import namedtuple

import dunlin.utils          as ut
import dunlin.datastructures as dst
import dunlin.ode.extrafunc  as exf

model_args_ = 'time', 'states', 'parameters'
model_args  = ', '.join(model_args_)

###############################################################################
#Top Level Algorithm
###############################################################################
ode_tup = namedtuple('ODE', 'rhs rhsdct rhsevents rhsextra')

def make_ode(model_data: dict) -> tuple[callable, callable, callable]:
    rhs       = make_rhs(model_data)
    rhsdct    = make_rhsdct(model_data)
    
    
    if model_data.get('events'):
        rhsevents = make_rhsevents(model_data)
    else:
        rhsevents = {}
    
    if 'extra' in model_data:
        rhsextra = make_rhsextra(model_data)
        # if callable(model_data['extra']):
        #     rhsextra = model_data['extra']
        # else:
        #     rhsextra = make_rhsextra(model_data)
        
    else:
        rhsextra = None

    return ode_tup(rhs, rhsdct, rhsevents, rhsextra)

###############################################################################
#Function Generation Algorithm
###############################################################################
def make_rhs(model_data):
    code, rhs_name = make_rhs_code(model_data)
    dct            = ut.code2func(code, rhs_name)
    rhs            = dct[rhs_name]
    
    test_t = 1
    test_y = np.ones(len(model_data['states']))
    test_p = np.ones(len(model_data['parameters']))
    
    test_func(rhs, test_t, test_y, test_p)
    
    code = '@__njit\n' + code
    dct            = ut.code2func(code, rhs_name)
    rhs            = dct[rhs_name]
    rhs.code       = code
    
    return rhs

def make_rhsdct(model_data):
    code, rhs_name = make_rhsdct_code(model_data)
    dct            = ut.code2func(code, rhs_name)
    rhs            = dct[rhs_name]
    
    test_t = 1
    test_y = np.ones((len(model_data['states']), 10))
    test_p = np.ones((len(model_data['parameters']), 10))
    
    test_func(rhs, test_t, test_y, test_p)
    
    code           = code
    dct            = ut.code2func(code, rhs_name)
    rhs            = dct[rhs_name]
    rhs.code       = code
    
    to_return = [*model_data['states'], 
                 *model_data['parameters'],
                 *model_data['variables'],
                 *model_data['reactions'], 
                 *[ut.diff(x) for x in model_data['states']]
                 ]
    
    rhs.variables = to_return
    
    return rhs

def test_func(func, *args, **kwargs):
    try:
        return func( *args, **kwargs)
    except NameError as e:
        e.args = (ut.dot(e.args[0]),)
        raise e
    except Exception:
        pass

###############################################################################
#Top Level Code Generation
###############################################################################
def make_rhs_code(model_data, undot=True):
    rhs_name   = 'model_' + model_data['ref']
    definition = ut.def_func(rhs_name, model_args)
    
    #Generate code
    code = [definition,
            xps2code(model_data),
            funcs2code(model_data), 
            vrbs2code(model_data), 
            rxnrts2code(model_data),
            make_model_return(model_data)
            ]
    
    code = '\n\n'.join(code)
    if undot:
        code = ut.undot(code)
    
    return code, rhs_name

def make_rhsdct_code(model_data):
    rhs_name   = 'modeldct_' + model_data['ref']
    outer      = ut.def_func(rhs_name, model_args)
    definition = outer
    
    #Generate code
    code = [xps2code(model_data),
            funcs2code(model_data), 
            vrbs2code(model_data), 
            rxnrts2code(model_data),
            ]
    
    code = '\n\n'.join(code)
    code = definition + '\n' + code
    code = ut.undot(code)
    
    code += '\n\n' + make_modeldct_return(model_data)
    
    return code, rhs_name

###############################################################################
#Model Definition and Return Value for INTEGRATION
###############################################################################
def make_model_return(model_data):
    dy = [ut.diff(x) for x in model_data['states']]
    dy = ', '.join(dy)
    
    return_value = f'\treturn __np.array([{dy}])'
    
    return return_value

def make_modeldct_return(model_data):
    #Do not use for submodels
    to_return = [*model_data['states'], 
                 *model_data['parameters'],
                 *model_data['variables'],
                 *model_data['reactions'], 
                 *[ut.diff(x) for x in model_data['states']]
                 ]
    
    # to_return    = [ut.undot(x) for x in to_return]
    return_value = '\treturn {' + ', '.join([f'"{x}": {ut.undot(x)} ' for x in to_return]) + '}'
    
    return return_value

###############################################################################
#Code generation for rhs
###############################################################################
def update_namespace(namespace, *to_add, unique=True):
    for i in to_add:
        if i in namespace and unique:
            raise ValueError(f'Repeated namespace: {i}')
        namespace.add(i)
        
def xps2code(model_data):
    global model_args_
    #Get the signature
    t, states, parameters = model_args_
    
    #Set up namespace cache
    xs   = model_data[states]
    ps   = model_data[parameters]
    code = ''
    
    if xs:
        code += '\t#States\n'# + xs.to_py()
        temp  = ['\t' + f'{name} = states[{i}]' for i, name in enumerate(xs)]
        code += '\n'.join(temp)
    
    if ps:
        code += '\n\t#Parameters\n' 
        temp  = ['\t' + f'{name} = parameters[{i}]' for i, name in enumerate(ps)]
        code += '\n'.join(temp)
        
    return code

def funcs2code(model_data):
    funcs = model_data.get('functions')
    
    if funcs:
        code = '\t#Functions\n' #+ funcs.to_py()
        for name, func in funcs.items():
            call  = f'{func.name}({func.signature})'
            expr  = f'return {func.expr}'
            code += f'\tdef {call}:\n\t\t{expr}\n'
    else:
        code = ''
        
    return code

def vrbs2code(model_data):
    vrbs = model_data.get('variables')
    
    if vrbs:
        code = '\t#Variables\n'
        for name, vrb in vrbs.items():
            code += f'\t{vrb.name} = {vrb.expr}\n'
    else:
        code = ''
        
    return code

def rxnrts2code(model_data):
    #Get the dct of differentials
    diffs1, rxns_code = _rxns2code(model_data)
    diffs2            = _rts2code(model_data)
    diffs             = {**diffs1, **diffs2}
    
    #Check
    repeat = set(diffs1).intersection(diffs2)
    if repeat:
        msg  = f'Error in model {model_data["ref"]}\n'
        msg += f'One or more states appeared in both a rate and reaction: {repeat}'
        raise ValueError(msg)
    
    missing = set(model_data['states']).difference(diffs)
    if missing:
        msg  =  f'Error in model {model_data["ref"]}\n'
        msg += f'One or states is not involved in a rate or reaction: {missing}'
        raise ValueError(msg)
        
    #Convert to strings
    diffs  = '\n'.join(diffs.values())
    diffs  = '\t#Differentials\n' + diffs
    
    #Merge with reactions
    diffs = rxns_code + '\n' + diffs
    
    return diffs

def _rts2code(model_data):
    rts   = model_data.get('rates')
    diffs = {}
    if rts:
        diffs = {state: f'\t{rt.name} = {rt.expr}'  for state, rt in rts.items()}

    return diffs

def _rxns2code(model_data):
    rxns  = model_data.get('reactions')
    diffs = {}
    
    if rxns:
        code = '\t#Reactions\n' 
        diffs = {}
        
        for name, rxn in rxns.items():
            code += f'\t{rxn.name} = {rxn.rate}\n'
            #Create the differentials
            for state, n in rxn.stoichiometry.items():
                diffs.setdefault(state, '\t' + ut.diff(state) + ' = ')
                diffs[state] += f'{n}*{name}'
                
    else:
        code = ''
       
    return diffs, code

###############################################################################
#Function and Code Generation for Events
###############################################################################
evs_tup = namedtuple('rhsevent', 'ref name trigger_func assign_func delay persistent priority')

def make_rhsevents(model_data):
    ref = model_data['ref']
    
    #Create cached values
    # xps_code  = xps2code(model_data)
    mid_code = [xps2code(model_data),
                funcs2code(model_data), 
                vrbs2code(model_data), 
                rxnrts2code(model_data),
                ]
    
    mid_code = '\n\n'.join(mid_code)
    
    rhsevents = {}
    
    #Make test values
    test_t = 1
    test_y = np.ones(len(model_data['states']))
    test_p = np.ones(len(model_data['parameters']))
    
    evs = model_data['events']
    #Generate function and code
    for name, ev in evs.items():
        trigger_name, trigger_code = trigger2code(ev, mid_code, model_data)
        assign_name, assign_code   = assign2code(ev, mid_code, model_data)
        trigger_name, assign_name  = ut.undot([trigger_name, assign_name])
        
        code = trigger_code + '\n\n' + assign_code
        code = ut.undot(code)
        dct  = ut.code2func(code, trigger_name, assign_name)
        
        trigger_func      = dct[trigger_name]
        assign_func       = dct[assign_name]
        trigger_func.code = trigger_code
        assign_func.code  = assign_code
        
        test_func(trigger_func, test_t, test_y, test_p)
        test_func(assign_func, test_t, test_y, test_p)
        rhsevents[name] = evs_tup(ref, name, trigger_func, assign_func,
                                  ev.delay, ev.persistent, ev.priority
                                  )
        
    return rhsevents
    
def trigger2code(ev, mid_code, model_data):
    global model_args
    
    ref        = model_data['ref']
    func_name  = f'trigger_{ref}_{ev.name}'
    func_def   = ut.def_func(func_name, model_args)
    return_val = '\treturn __triggered'
    trigger    = f'\t__triggered = {ev.trigger_expr}'
    trigger    = '\t#Trigger\n' + trigger
    code       = [func_def, mid_code, trigger, return_val ]
    code       = '\n\n'.join(code)

    return func_name, code

def assign2code(ev, mid_code, model_data):
    global _args
    
    ref        = model_data['ref']
    func_name  = f'assign_{ref}_{ev.name}'
    func_def   = ut.def_func(func_name, model_args)
    assign     = '\n'.join(['\t' + i for i in ev.assign_expr])
    assign     = '\t#Assign\n' + assign
    xs         = model_data['states']
    ps         = model_data['parameters']
    new_y      = f'\tnew_y = __np.array([{", ".join(xs)}])'
    new_p      = f'\tnew_p = __np.array([{", ".join(ps)}])'
    collate    = '\n'.join(['\t#Collate', new_y, new_p])
    return_val = '\treturn new_y, new_p'
    code       = '\n\n'.join([func_def, 
                              mid_code, 
                              assign, 
                              collate,
                              return_val
                              ])
    
    return func_name, code
    
###############################################################################
#Extra
###############################################################################
def make_rhsextra(model_data):
    rhs_name, code = make_rhsextra_code(model_data)
    globalvars     = {**ut.default_globals, **exf.exf}
    dct            = ut.code2func(code, rhs_name, globalvars=globalvars)
    rhs            = dct[rhs_name]

    rhs.code  = code
    rhs.names = list(model_data['extra'])
    return rhs

def make_rhsextra_code(model_data):
    rhs_name   = 'extra_' + model_data['ref']
    definition = ut.def_func(rhs_name, model_args)
    
    #Make template
    template = [definition,
                xps2code(model_data),
                funcs2code(model_data), 
                vrbs2code(model_data), 
                rxnrts2code(model_data),
                ]
    template = '\n\n'.join(template)
    
    #Generate assignments
    assign     = exs2code(model_data)
    return_val = ', '.join([f'"{i}": {ut.undot(i)}' for i in model_data['extra']])
    return_val = '\treturn {' + return_val + '}'
    
    #Join everything
    code = '\n\n'.join([template, assign])
    code = ut.undot(code)
    code += '\n\n' + return_val
    
    return rhs_name, code
    
def exs2code(model_data):
    #Set up caches
    extras = model_data['extra']
    code   = '\t#Extra\n' 
    
    for name, ex in extras.items():
        code += f'\t{ex.name} = __{ex.func_name}({ex.signature})\n'

    return code

  