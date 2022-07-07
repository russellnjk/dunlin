import addpath
import dunlin.comp as cmp
from dunlin.ode.code_rhs import *
from data import all_data

model_data = cmp.make_model_data(all_data, 'M0')

xp_code     = xps2code(model_data)
funcs_code  = funcs2code(model_data)
vrb_code    = vrbs2code(model_data)
rxnrts_code = rxnrts2code(model_data)

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

r = rhs(test_t, test_y, test_p)
# print(r)

#Try on hierarchical model
model_data = cmp.flatten(all_data, 'M1')
code, rhs_name = make_rhs_code(model_data)

dct            = ut.code2func(code, rhs_name)
rhs            = dct[rhs_name]

test_t = 1
test_y = np.ones(len(model_data['states']))
test_p = np.ones(len(model_data['parameters']))

r = rhs(test_t, test_y, test_p)
# print(r)

rhs = make_rhs(model_data)

#Test rhsdct
model_data = cmp.flatten(all_data, 'M0')
rhsdct     = make_rhsdct(model_data)

r = rhsdct(test_t, test_y, test_p)
print(r)


