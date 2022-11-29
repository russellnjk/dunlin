import addpath
import dunlin                            as dn
import dunlin.utils                      as ut
import dunlin.standardfile.dunl          as sfd
from dunlin.datastructures.modeldata_backup import ModelData
from data import *

class TestClass(ModelData):
    _attrs = {'name'   : [str],
              'string' : [str, None],
              'number' : [int, 1.5]
              }
    
#Test assignment
tc0 = TestClass()

tc0.name = 'A'
assert tc0.name == 'A'

tc0.string = None
assert tc0.string == None

tc0.number = 3
assert tc0.number == 3

tc0.number = 1.5
assert tc0.number == 1.5

try:
    tc0.string = 3
except:
    assert True
else:
    assert False

try:
    tc0.b= 3
except:
    assert True
else:
    assert False
    
#Test export to Python
dct = tc0.to_data()

assert dct == {'name': 'A', 'string': None, 'number': 1.5}

# print(dct)

#Test export to dunl
dunl = sfd.write_dunl_code({tc0.name: tc0})
a    = sfd.read_dunl_code(dunl)
# assert a[ref] == d[ref]
