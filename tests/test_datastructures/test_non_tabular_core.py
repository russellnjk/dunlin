import addpath
import dunlin                            as dn
import dunlin.utils                      as ut
import dunlin.standardfile.dunl.readdunl as rdn
from dunlin.datastructures.function import Function, FunctionDict
from dunlin.datastructures.reaction import Reaction, ReactionDict
from dunlin.datastructures.variable import Variable, VariableDict
from dunlin.datastructures.rate     import Rate,     RateDict
from dunlin.datastructures.extra    import ExtraVariable, ExtraDict
from dunlin.datastructures.event    import Event,    EventDict

###############################################################################
#Test function
###############################################################################
data0 = {'f0': ['x', 'y', 'x+y'],
         'f1': ['y', 'y-1']
         }
ext_namespace = ()
C = FunctionDict

#Test instantiation
F0 = C(data0, set(ext_namespace))

try:
    F0 = C(data0, ext_namespace=set(ext_namespace + ('f0',)))
except:
    assert True
else:
    assert False
    
#Test access
f0 = F0['f0']

#Test export/roundtrip
data1 = F0.to_data()
dunl = F0.to_dunl()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0

###############################################################################
#Test reaction
###############################################################################
data0 = {'f0' : ['a -> b', 'k0*a', 'k1*b', (-10, 10)],
         'f1' : {'eqn' : 'b -> c', 'fwd' : 'k2*b'}
         }
ext_namespace = ('a', 'b', 'c', 'k0', 'k1', 'k2')
C = ReactionDict

#Test instantiation
F0 = C(data0, ext_namespace=set(ext_namespace))
try:
    F0 = C(data0, ext_namespace=set(ext_namespace[1:]))
except:
    assert True
else:
    assert False

try:
    F0 = C(data0, ext_namespace=set(ext_namespace + ('f0',)))
except:
    assert True
else:
    assert False

#Test access
f0 = F0['f0']

#Test export/roundtrip
data1 = F0.to_data()
dunl = F0.to_dunl()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2['f1'] == data1['f1'] == data0['f1']

##############################################################################
#Test variable
##############################################################################
data0 = {'f0': 1,
         'f1': '(x + y)/2'
         }
ext_namespace = ('x', 'y')
C = VariableDict

F0 = C(data0, set(ext_namespace))
try:
    F0 = C(data0, ext_namespace=set(ext_namespace[1:]))
except:
    assert True
else:
    assert False

try:
    F0 = C(data0, ext_namespace=set(ext_namespace + ('f0',)))
except:
    assert True
else:
    assert False
    
f0 = F0['f0']

#Test export/roundtrip
data1 = F0.to_data()
dunl = F0.to_dunl()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0

##############################################################################
#Test rate
##############################################################################
data0 = {'f0': '-k0*f0',
         'f1': '-k1*f1'
         }
ext_namespace = ('f0', 'f1', 'k0', 'k1')
C = RateDict

F0 = C(data0, set(ext_namespace))
try:
    F0 = C(data0, ext_namespace=set(ext_namespace[1:]))
except:
    assert True
else:
    assert False

try:
    F0 = C(data0, ext_namespace=set(ext_namespace + (ut.diff('f0'),)))
except:
    assert True
else:
    assert False
    
f0 = F0['f0']

#Test export/roundtrip
data1 = F0.to_data()
dunl = F0.to_dunl()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0

##############################################################################
#Test rate
##############################################################################
data0 = {'f0': '-k0*f0',
         'f1': '-k1*f1'
         }
ext_namespace = ('f0', 'f1', 'k0', 'k1')
C = RateDict

F0 = C(data0, set(ext_namespace))
try:
    F0 = C(data0, ext_namespace=set(ext_namespace[1:]))
except:
    assert True
else:
    assert False

try:
    F0 = C(data0, ext_namespace=set(ext_namespace + (ut.diff('f0'),)))
except:
    assert True
else:
    assert False
    
f0 = F0['f0']

#Test export/roundtrip
data1 = F0.to_data()
dunl = F0.to_dunl()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0

##############################################################################
#Test extra
##############################################################################
data0 = {'f0': ['index', 'x0', -1],
         'f1': ['index', 'x1',  0]
         }
ext_namespace = ('x0', 'x1')
C = ExtraDict

F0 = C(data0, set(ext_namespace))
try:
    F0 = C(data0, ext_namespace=set(ext_namespace[1:]))
except:
    assert True
else:
    assert False

try:
    F0 = C(data0, ext_namespace=set(ext_namespace + ('f0',)))
except:
    assert True
else:
    assert False
    
f0 = F0['f0']

#Test export/roundtrip
data1 = F0.to_data()
dunl = F0.to_dunl()
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

ext_namespace = ('x0', 'x1')
C = EventDict

F0 = C(data0, set(ext_namespace))
try:
    F0 = C(data0, ext_namespace=set(ext_namespace[1:]))
except:
    assert True
else:
    assert False

try:
    F0 = C(data0, ext_namespace=set(ext_namespace + ('f0',)))
except:
    assert True
else:
    assert False
    
f0 = F0['f0']

#Test export/roundtrip
data1 = F0.to_data()
dunl = F0.to_dunl()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0

