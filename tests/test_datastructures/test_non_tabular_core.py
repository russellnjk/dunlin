import addpath
import dunlin                            as dn
import dunlin.utils                      as ut
import dunlin.standardfile.dunl.readdunl as rdn
from dunlin.datastructures.function   import Function, FunctionDict
from dunlin.datastructures.reaction   import Reaction, ReactionDict
from dunlin.datastructures.variable   import Variable, VariableDict
from dunlin.datastructures.rate       import Rate,     RateDict
from dunlin.datastructures.event      import Event,    EventDict
from dunlin.datastructures.stateparam import StateDict, ParameterDict

###############################################################################
#Test function
###############################################################################
data0 = {'f0': ['x', 'y', 'x+y'],
         'f1': ['y', 'y-1']
         }
all_names = ()
C = FunctionDict

#Test instantiation
F0 = C(set(all_names), data0)

try:
    F0 = C(set(all_names + ('f0',)), data0)
except:
    assert True
else:
    assert False
    
#Test access
f0 = F0['f0']

#Test export/roundtrip
data1 = F0.to_dict()
dunl = F0.to_dunl_elements()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0

###############################################################################
#Test reaction
###############################################################################
data0 = {'f0' : ['a -> b', 'k0*a - k1*b', [-10, 10]],
         'f1' : {'equation' : 'b -> c', 'rate' : 'k2*b'}
         }

all_names = {'k0', 'k1', 'k2'}
C = ReactionDict
xs = StateDict(all_names, {'a': [0], 'b': [0], 'c': [0]})

#Test instantiation
F0 = C(set(all_names), xs, data0)
try:
    F0 = C(set(all_names[1:]), data0)
except:
    assert True
else:
    assert False

try:
    F0 = C(set(all_names + ('f0',)), data0)
except:
    assert True
else:
    assert False

#Test access
f0 = F0['f0']

#Test export/roundtrip
data1 = F0.to_dict()
dunl = F0.to_dunl_elements()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2['f1'] == data1['f1'] == data0['f1']

##############################################################################
#Test variable
##############################################################################
data0 = {'f0': 1,
         'f1': '(x + y)/2'
         }
all_names = ('x', 'y')
C = VariableDict

F0 = C(set(all_names), data0)
try:
    F0 = C(set(all_names[1:]), data0)
except:
    assert True
else:
    assert False

try:
    F0 = C(set(all_names + ('f0',)), data0)
except:
    assert True
else:
    assert False
    
f0 = F0['f0']

#Test export/roundtrip
data1 = F0.to_dict()
dunl = F0.to_dunl_elements()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0

##############################################################################
#Test rate
##############################################################################
data0 = {'f0': '-k0*f0',
         'f1': '-k1*f1'
         }
all_names = set()
C = RateDict

states = StateDict(all_names, {'f0': [0], 'f1': [0]})
all_names.update(['k0', 'k1']) 

F0 = C(set(all_names), states, data0)
try:
    F0 = C(set(all_names[1:]),data0)
except:
    assert True
else:
    assert False

try:
    F0 = C(set(all_names + (ut.diff('f0'),)), data0)
except:
    assert True
else:
    assert False
    
f0 = F0['f0']

#Test export/roundtrip
data1 = F0.to_dict()
dunl = F0.to_dunl_elements()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0

##############################################################################
#Test event
##############################################################################
data0 = {'f0': {'trigger'    : 'time > 10',
                'assign'     : 'x0 = 2',
                'delay'      : 5,
                },
         'f1': {'trigger'    : 'time > 20',
                'assign'     : ['x1 = 2', 'x0 = 3'],
                'delay'      : 5,
                'persistent' : False,
                'priority'   : 1
                }
         }

all_names = {'x0', 'x1'}
C = EventDict
F0 = C(set(all_names), data0)
try:
    F0 = C(set(all_names[1:]), data0)
except:
    assert True
else:
    assert False

try:
    F0 = C(set(all_names + ('f0',)), data0)
except:
    assert True
else:
    assert False
    
f0 = F0['f0']

#Test export/roundtrip
data1 = F0.to_dict()
dunl = F0.to_dunl_elements()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0

