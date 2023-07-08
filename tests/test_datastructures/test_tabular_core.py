import addpath
import dunlin                            as dn
import dunlin.standardfile.dunl.readdunl as rdn
from dunlin.datastructures.stateparam import StateDict, ParameterDict

###############################################################################
#Test states
###############################################################################
data0 = {'x0': {'c0' : 0, 'c1': 1},
         'x1': {'c0' : 0, 'c1': 1},
         'x2': {'c0' : 0, 'c1': 1},
         }
all_names = set()
C = StateDict

#Test instantiation
F0 = C(set(all_names), data0)

try:
    F0 = C(data0, all_names=set(all_names + ('x0',)))
except:
    assert True
else:
    assert False
    
#Test access
x0 = F0.df['x0']

#Test export/roundtrip
data1 = F0.to_dict()
dunl = F0.to_dunl_elements()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data0

###############################################################################
#Test parameters
###############################################################################
data0 = {'x0': {'c0' : 0, 'c1': 1},
         'x1': {'c0' : 0, 'c1': 1},
         'x2': {'c0' : 0, 'c1': 1},
         }
all_names = set()
C = ParameterDict

#Test instantiation
F0 = C(set(all_names), data0)

try:
    F0 = C(data0, all_names=set(all_names + ('x0',)))
except:
    assert True
else:
    assert False
    
#Test access
x0 = F0.df['x0']

#Test export/roundtrip
data1 = F0.to_dict()
dunl = F0.to_dunl_elements()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data0
