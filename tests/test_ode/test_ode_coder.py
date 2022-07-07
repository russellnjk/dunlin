import numpy as np

import addpath
import dunlin.utils as ut
import dunlin.comp as cmp
import dunlin.datastructures as dst
from dunlin.ode.ode_coder import *
from data import all_data

model_data = cmp.make_model_data(all_data, 'M0')
model_data = dst.ODEModelData(**model_data)

xp_code     = xps2code(model_data)
funcs_code  = funcs2code(model_data)
vrb_code    = vrbs2code(model_data)
rxnrts_code = rxnrts2code(model_data)

# print(xp_code)
# print(funcs_code)
# print(vrb_code)
# print(rxnrts_code)

code, rhs_name = make_rhs_code(model_data)
# print(code)

dct            = ut.code2func(code, rhs_name)
rhs            = dct[rhs_name]

test_t = 1
test_y = np.ones(len(model_data['states']))
test_p = np.ones(len(model_data['parameters']))

dy = rhs(test_t, test_y, test_p)
assert np.all(dy == [-1, 2, -1, -1])

rhs = make_rhs(model_data)
dy = rhs(test_t, test_y, test_p)
assert np.all(dy == [-1, 2, -1, -1])

###############################################################################
#Test Event Code/Function Generation
###############################################################################
#Set up
mid_code = [xps2code(model_data),
            funcs2code(model_data), 
            vrbs2code(model_data), 
            rxnrts2code(model_data),
            ]

mid_code = '\n\n'.join(mid_code)
name     = 'ev0'
ev       = model_data['events']['ev0']

test_t = 0
test_y = np.zeros(len(model_data['states']))
test_p = np.zeros(len(model_data['parameters']))

#Test trigger
trigger_name, trigger_code= trigger2code(ev, mid_code, model_data)
# print(trigger_code)

dct          = ut.code2func(trigger_code, trigger_name)
trigger_func = dct[trigger_name]

triggered = trigger_func(test_t, test_y, test_p)
assert triggered == 0

#Test assign
assign_name, assign_code = assign2code(ev, mid_code, model_data)
# print(assign_code)

dct         = ut.code2func(assign_code, assign_name)
assign_func = dct[assign_name]

new_y, new_p = assign_func(test_t, test_y, test_p)
assert np.all(new_y == np.array([0, 0, 1, 1]))
assert np.all(new_p == test_p)

#Test one-step function generation
rhsevents = make_rhsevents(model_data)
rhsevent  = rhsevents['ev0']

trigger_func = rhsevent.trigger_func
assign_func  = rhsevent.assign_func

triggered = trigger_func(test_t, test_y, test_p)
assert triggered == 0

new_y, new_p = assign_func(test_t, test_y, test_p)
assert np.all(new_y == np.array([0, 0, 1, 1]))
assert np.all(new_p == test_p)

###############################################################################
#Test Extra Variables
###############################################################################
#Set up
model_data  = cmp.make_model_data(all_data, 'M0')
extra       = {'ex0': ['index', 'x0', '-1'],
               'ex1': ['where', 'x1>5', 'time']
               }
model_data_ = {**model_data, **{'extra': extra}}
model_data_ = dst.ODEModelData(**model_data_)

test_t = np.linspace(0, 10, 11)
test_y = np.array([np.linspace(0, 20, 11) for i in model_data_['states']])
test_p = np.array([np.linspace(0, 20, 11) for i in model_data_['parameters']])

#Test extra
extra_code = exs2code(model_data_)
# print(extra_code)

rhs_name, rhsextra_code = make_rhsextra_code(model_data_)
# print(rhsextra_code)

rhsextra = make_rhsextra(model_data_)
exs      = rhsextra(test_t, test_y, test_p)
# print(exs)
assert exs == {'ex0': 20, 'ex1': 3}

###############################################################################
#Test High Level
###############################################################################
model_data = cmp.make_model_data(all_data, 'M0')
extra      = {'ex0': ['index', 'x0', '-1'],
              'ex1': ['where', 'x1>5', 'time']
              }

model_data['extra'] = extra
model_data          = dst.ODEModelData(**model_data)

ode = make_ode(model_data)
rhs, rhsdct, rhsevents, rhsextra = ode

#Test rhs 
test_t = 1
test_y = np.ones(len(model_data['states']))
test_p = np.ones(len(model_data['parameters']))

dy = ode.rhs(test_t, test_y, test_p)
assert np.all(dy == [-1, 2, -1, -1])

#Test events
test_t = 0
test_y = np.zeros(len(model_data['states']))
test_p = np.zeros(len(model_data['parameters']))

rhsevent  = ode.rhsevents['ev0']

trigger_func = rhsevent.trigger_func
assign_func  = rhsevent.assign_func

triggered = trigger_func(test_t, test_y, test_p)
assert triggered == 0

new_y, new_p = assign_func(test_t, test_y, test_p)
assert np.all(new_y == np.array([0, 0, 1, 1]))
assert np.all(new_p == test_p)

#Test dct and extra
test_t = np.linspace(0, 10, 11)
test_y = np.array([np.linspace(0, 20, 11) for i in model_data_['states']])
test_p = np.array([np.linspace(0, 20, 11) for i in model_data_['parameters']])

dct = ode.rhsdct(test_t, test_y, test_p)
# print(dct)

exs = ode.rhsextra(test_t, test_y, test_p)
# print(exs)
assert exs == {'ex0': 20, 'ex1': 3}

###############################################################################
#Hierarchical Model
###############################################################################
#Try on hierarchical model
model_data = cmp.flatten(all_data, 'M1')
extra      = {'ex0': ['index', 'x0', '-1'],
               'ex1': ['where', 'm0.x1>5', 'time']
               }
model_data['extra'] = extra
model_data          = dst.ODEModelData(**model_data)

xp_code     = xps2code(model_data)
funcs_code  = funcs2code(model_data)
vrb_code    = vrbs2code(model_data)
rxnrts_code = rxnrts2code(model_data)

# print(funcs_code)
# print(vrb_code)
# print(rxnrts_code)

code, rhs_name = make_rhs_code(model_data)
code, rhs_name = make_rhsdct_code(model_data)
# print(code)

ode        = make_ode(model_data)

#Test rhs
test_t = 1
test_y = np.ones(len(model_data['states']))
test_p = np.ones(len(model_data['parameters']))

dy = ode.rhs(test_t, test_y, test_p)
# print(dy)
assert np.all(dy == [-1, 2, -1, -1, -0.1])

#Test events
test_t = 0
test_y = np.zeros(len(model_data['states']))
test_p = np.zeros(len(model_data['parameters']))

rhsevent  = ode.rhsevents['m0.ev0']

trigger_func = rhsevent.trigger_func
assign_func  = rhsevent.assign_func

triggered = trigger_func(test_t, test_y, test_p)
assert triggered == 0

#Test dct and extra
test_t = np.linspace(0, 10, 11)
test_y = np.array([np.linspace(0, 20, 11) for i in model_data['states']])
test_p = np.array([np.linspace(0, 20, 11) for i in model_data['parameters']])

dct = ode.rhsdct(test_t, test_y, test_p)
# print(dct)

exs = ode.rhsextra(test_t, test_y, test_p)
# print(exs)
assert exs == {'ex0': 20, 'ex1': 3}
