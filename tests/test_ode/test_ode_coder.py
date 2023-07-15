import numpy as np

import addpath
import dunlin.utils as ut
import dunlin.comp as cmp
import dunlin.datastructures as dst
from dunlin.datastructures import ODEModelData
from dunlin.ode.ode_coder import *
from data import all_data

ode_data = ODEModelData.from_all_data(all_data, 'M0')

state_code     = states2code(ode_data)
parameter_code = parameters2code(ode_data)
functions_code = functions2code(ode_data)
variables_code = variables2code(ode_data)
rate_code      = rates2code(ode_data)
reactionscode  = reactions2code(ode_data)

###############################################################################
#Test rhs
###############################################################################
rhs, _  = make_rhs(ode_data)
code    = rhs.code 
# print(code)

test_t = 1
test_y = np.ones(len(ode_data.states))
test_p = np.ones(len(ode_data.parameters))

dy = rhs(test_t, test_y, test_p)
assert np.all(dy == [-1, 2, -1, -1])

###############################################################################
#Test Event Code/Function Generation
###############################################################################
events           = make_events(ode_data)
event            = events[0]
trigger_functiontion = event.trigger_function
assignment_function = event.assignment_function

test_t = 0
test_y = np.zeros(len(ode_data.states))
test_p = np.zeros(len(ode_data.parameters))

#Test trigger
triggered = trigger_functiontion(test_t, test_y, test_p)
assert triggered == 0

#Test assignment
new_y, new_p = assignment_function(test_t, test_y, test_p)
assert np.all(new_y == np.array([0, 0, 1, 1]))
assert np.all(new_p == test_p)

###############################################################################
#Test rhsdct
###############################################################################
rhsdct, _  = make_rhsdct(ode_data)
code    = rhsdct.code 
# print(code)

test_t = np.array([0, 1])
test_y = np.ones((len(ode_data.states), 2))
test_p = np.ones((len(ode_data.parameters), 2))

dct = rhsdct(test_t, test_y, test_p)
assert np.all(dct['x0'] == [1, 1])
assert np.all(dct[ut.diff('x0')] == [-1, -1])

###############################################################################
#Test High Level
###############################################################################
ode_data = ODEModelData.from_all_data(all_data, 'M0')

(rhs, _), (rhsdct, _), events = make_ode_callables(ode_data)

#Test rhs 
test_t = 1
test_y = np.ones(len(ode_data.states))
test_p = np.ones(len(ode_data.parameters))

dy = rhs(test_t, test_y, test_p)
assert np.all(dy == [-1, 2, -1, -1])

#Test events
event            = events[0]
trigger_functiontion = event.trigger_function
assignment_function = event.assignment_function

test_t = 0
test_y = np.zeros(len(ode_data.states))
test_p = np.zeros(len(ode_data.parameters))

#Test trigger
triggered = trigger_functiontion(test_t, test_y, test_p)
assert triggered == 0

#Test assignment
new_y, new_p = assignment_function(test_t, test_y, test_p)
assert np.all(new_y == np.array([0, 0, 1, 1]))
assert np.all(new_p == test_p)

#Test dct
test_t = np.array([0, 1])
test_y = np.ones((len(ode_data.states), 2))
test_p = np.ones((len(ode_data.parameters), 2))

dct = rhsdct(test_t, test_y, test_p)
assert np.all(dct['x0'] == [1, 1])
assert np.all(dct[ut.diff('x0')] == [-1, -1])

###############################################################################
#Hierarchical Model
###############################################################################
#Try on hierarchical model
ode_data = ODEModelData.from_all_data(all_data, 'M1')

(rhs, _), (rhsdct, _), events = make_ode_callables(ode_data)

#Test rhs
test_t = 1
test_y = np.ones(len(ode_data.states))
test_p = np.ones(len(ode_data.parameters))

dy = rhs(test_t, test_y, test_p)
# print(dy)
assert np.all(dy == [-1, 2, -1, -1, -0.1])

#Test events
test_t = 0
test_y = np.zeros(len(ode_data.states))
test_p = np.zeros(len(ode_data.parameters))

event = events[0]

trigger_function = event.trigger_function
assign_func  = event.assignment_function

triggered = trigger_function(test_t, test_y, test_p)
assert triggered == 0

#Test dct and extra
test_t = np.linspace(0, 10, 11)
test_y = np.array([np.linspace(0, 20, 11) for i in ode_data.states])
test_p = np.array([np.linspace(0, 20, 11) for i in ode_data.parameters])

dct = rhsdct(test_t, test_y, test_p)
# print(dct)
