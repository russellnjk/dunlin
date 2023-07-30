import numpy as np
from collections import namedtuple
from numba       import njit
from typing      import Callable

import dunlin.utils          as ut
import dunlin.datastructures as dst
from dunlin.datastructures import ODEModelData
from dunlin.ode.event      import Event

signature = 'time', 'states', 'parameters'
model_args  = ', '.join(signature)

###############################################################################
#Code Generation for Integration
###############################################################################
def states2code(ode_data: ODEModelData, array:bool=False) -> str:
    global signature
    #Get the signature
    t, states, parameters = signature
    
    #Make code for states
    state_names = ode_data.states
    state_code  = '\t#States\n'
    diff_code   = '\t#Differentials\n'
    
    for i, state_name in enumerate(state_names):
        state_code  += f'\t{state_name} = states[{i}]\n'
        if array:
            diff_code   += f'\t{ut.diff(state_name)} = __zeros(len({state_name}))\n'
        else:
            diff_code   += f'\t{ut.diff(state_name)} = 0\n'
    
    state_code += '\n' + diff_code
    state_code  = ut.undot(state_code) 
    return state_code

def parameters2code(ode_data: ODEModelData) -> str:
    #Make code for parameters
    parameter_names = ode_data.parameters
    parameter_code  = '\t#Parameters\n'
    
    for i, parameter_name in enumerate(parameter_names):
        parameter_code += f'\t{parameter_name} = parameters[{i}]\n'
    
    parameter_code = ut.undot(parameter_code)
    return parameter_code

function_template = '\tdef {function_name}({signature}):\n\t\treturn {expr}'

def functions2code(ode_data: ODEModelData):
    global function_template
    
    functions     = ode_data.functions
    function_code = '\t#Functions\n'
    
    if not functions:
        return function_code
    
    for function_name, function in functions.items():
        signature      = ', '.join(function.signature)
        function_code += function_template.format(function_name = function_name,
                                                  signature     = signature,
                                                  expr          = function.expr
                                                  )
        
    function_code = ut.undot(function_code)
    return function_code

def variables2code(ode_data: ODEModelData) -> str:
    variables     = ode_data.variables
    variable_code = '\t#Variables\n'
    
    if not variables:
        return variable_code

    for variable_name, variable in variables.items():
        lhs = f'{variable_name}'
        rhs = f'{variable.expr}'
        
        variable_code += f'\t{lhs} = {rhs}\n'
    
    variable_code = ut.undot(variable_code)
    return variable_code

def rates2code(ode_data: ODEModelData) -> str:
    rates     = ode_data.rates
    rate_code = '\t#Rates\n'
    
    if not rates:
        return rate_code
    
    for state_name, rate in rates.items():
        lhs = f'{ut.diff(state_name)}'
        rhs = f'{rate.expr}'
        
        rate_code += f'\t{lhs} += {rhs}\n'
    
    rate_code = ut.undot(rate_code)
    return rate_code

def reactions2code(ode_data: ODEModelData) -> str:
    reactions     = ode_data.reactions
    reaction_code = '\t#Reactions\n'
      
    if not reactions:
        return reaction_code
    
    for reaction_name, reaction in reactions.items():
        rate = reaction.rate
        
        if reaction.bounds:
            lb, ub = reaction.bounds
            rate   = f'__min(ub, __max(lb, {rate}))'
            
        #Update reaction code
        reaction_code += f'\t{reaction_name} = {rate}\n' 

        #Update differentials
        for state_name, n in reaction.stoichiometry.items():
            lhs = f'{ut.diff(state_name)}'
            rhs = f'{n}*{reaction_name}'
            
            reaction_code += f'\t{lhs} += {rhs}\n' 
    
    reaction_code = ut.undot(reaction_code)
    return reaction_code

###############################################################################
#Code Generation for Events
###############################################################################
def trigger2code(event: dst.Event):
    
    trigger_code = f'\treturn {event.trigger}\n'
    
    trigger_code = ut.undot(trigger_code)
    return trigger_code

def assignment2code(event: dst.Event):
    assignment_code = '\t#Assignment\n'
    
    for lhs, rhs in event.assignments.items():
        assignment_code += f'\t{lhs} = {rhs}\n'
    
    assignment_code = ut.undot(assignment_code)
    return assignment_code

###############################################################################
#Function Generation
###############################################################################
rhs_functions    = {'__ndarray'     : np.ndarray,
                    '__array'       : np.array, 
                    '__float64'     : np.float64,
                    '__int32'       : np.int32,
                    '__min'         : np.minimum, 
                    '__max'         : np.maximum,
                    '__zeros'       : np.zeros,
                    '__ones'        : np.ones,
                    '__mean'        : np.mean,
                    '__median'      : np.median,
                    '__concatenate' : np.concatenate,
                    }
rhsdct_functions = rhs_functions.copy()

def make_rhs(ode_data):
    global rhs_functions
    global signature
    
    #Make function definition
    rhs_name   = 'model_' + ode_data.ref
    signature_ = ', '.join(signature)
    definition = f'def {rhs_name}({signature_}):'
    
    #Make return value
    d_states   = [ut.diff(ut.undot(x)) for x in ode_data.states]
    d_states   = ', '.join(d_states)
    return_val = f'\treturn __array([{d_states}])'
    
    #Make full code
    code = [definition,
            states2code(ode_data),
            parameters2code(ode_data),
            functions2code(ode_data), 
            variables2code(ode_data), 
            reactions2code(ode_data),
            rates2code(ode_data),
            return_val
            ]
    code = '\n\n'.join(code)
    
    #Execute
    scope = {}
    exec(code, rhs_functions, scope)
    
    #Extract
    rhs0      = scope[rhs_name]
    rhs0.code = code
    rhs1      = njit(rhs0)
    rhs1.code = code
    
    return rhs0, rhs1

def make_rhsdct(ode_data):
    global rhsdct_functions
    global signature
    
    #Make function definition
    rhs_name   = 'model_' + ode_data.ref
    signature_ = ', '.join(signature)
    definition = f'def {rhs_name}({signature_}):'
    
    #Make return value
    d_states   = [ut.diff(ut.undot(x)) for x in ode_data.states]
    d_states   = ', '.join(d_states)
    return_val = f'\treturn __array([{d_states}])'

    to_return  = [*ode_data.states, 
                  *ode_data.parameters,
                  *ode_data.variables,
                  *ode_data.reactions, 
                  *[ut.diff(x) for x in ode_data.states]
                  ]
    return_val = ut.undot(', '.join(to_return))
    return_val = f'\treturn {return_val}'
    
    #Make full code
    code = [definition,
            states2code(ode_data, array=True),
            parameters2code(ode_data),
            functions2code(ode_data), 
            variables2code(ode_data), 
            reactions2code(ode_data),
            rates2code(ode_data),
            return_val
            ]
    code = '\n\n'.join(code)
    
    #Execute
    scope = {}
    exec(code, rhsdct_functions, scope)
    
    #Extract
    rhsdct0      = _wrap(scope[rhs_name], to_return)
    rhsdct0.code = code
    rhsdct1      = _wrap(njit(scope[rhs_name]), to_return)
    rhsdct1.code = code
    
    return rhsdct0, rhsdct1

def _wrap(rhsdct: Callable, to_return: list[str]) -> Callable:
    def helper(time, states, parameters):
        tup = rhsdct(time, states, parameters)
        dct = dict(zip(to_return, tup))
        
        return dct
    return helper
        
def make_events(ode_data        : ODEModelData,
                trigger2code    : Callable = trigger2code,
                assignment2code : Callable = assignment2code,
                body_code       : str = None,
                new_y_code      : str = None 
                ) -> list[Event]:
    global signature
    global rhs_functions
    
    events        = ode_data.events
    event_objects = []
    
    if not events:
        return event_objects
    
    #When body_code is None, generate it here
    #Otherwise, use the code provided by the developer
    #This allows extensions such as spatial models
    if not body_code:
        body_code = [states2code(ode_data),
                      parameters2code(ode_data),
                      functions2code(ode_data), 
                      variables2code(ode_data), 
                      reactions2code(ode_data),
                      rates2code(ode_data),
                      ]
        body_code = '\n\n'.join(body_code)
    
    #Boilerplate top/front part of the code
    signature_ = ', '.join(signature)
    states     = ', '.join([ut.undot(x) for x in ode_data.states    ])
    parameters = ', '.join([ut.undot(p) for p in ode_data.parameters])
    
    for event_name, event in events.items():
        #Make trigger function
        trigger_code = trigger2code(event)
        trigger_name = f'trigger_{ut.undot(event_name)}'
        
        trigger_code = [f'def {trigger_name}({signature_}):',
                        body_code,
                        trigger_code
                        ]
        trigger_code = '\n\n'.join(trigger_code)
        
        scope = {}
        exec(trigger_code, rhs_functions, scope)
        
        trigger_function      = scope[trigger_name]
        trigger_function.code = trigger_code
        
        #Make assignment function
        assignment_code = assignment2code(event)
        assignment_name = f'assignment_{ut.undot(event_name)}'
        new_y           = new_y_code if new_y_code else f'\tnew_y = __array([{states}])'
        new_p           = f'\tnew_p = __array([{parameters}])' 
        assignment_code = [f'def {assignment_name}({signature_}):',
                           body_code,
                           assignment_code,
                           new_y,
                           new_p,
                           '\treturn new_y, new_p'
                           ]
        assignment_code = '\n\n'.join(assignment_code)
        
        scope = {}
        exec(assignment_code, rhs_functions, scope)
        
        assignment_function      = scope[assignment_name]
        assignment_function.code = assignment_code
        
        #Instantiate event
        event_object = Event(event_name, 
                             trigger_function, 
                             assignment_function, 
                             event.delay, 
                             event.persistent, 
                             event.priority, 
                             ode_data.ref
                             )
        #Update results
        event_objects.append(event_object)
    
    event_objects = sorted(event_objects, key=lambda event: event.priority)
    
    return event_objects

###############################################################################
#Top Level Algorithm
###############################################################################
ode_tup = namedtuple('ODE', 'rhs rhsdct events')

def make_ode_callables(ode_data: dict) -> tuple[tuple[Callable], tuple[Callable], tuple[Event]]:
    rhs    = make_rhs(ode_data)
    rhsdct = make_rhsdct(ode_data)
    events = make_events(ode_data)

    return ode_tup(rhs, rhsdct, events)  