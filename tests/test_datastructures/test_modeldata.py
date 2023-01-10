import addpath
import dunlin                        as dn
import dunlin.utils                  as ut
import dunlin.standardfile.dunl      as sfd
from dunlin.datastructures.modeldata import ModelData
from data import *

class TestClass(ModelData):
    def __init__(self, ref, a, b):
        self.ref = ref
        self.a   = a
        self.b   = b
    
    def to_data(self, recurse=False) -> dict:
        data = {self.ref: {'a': self.a,
                           'b': self.b
                           }
                }
        data = self._to_data(keys=['a', 'b'], recurse=recurse)
        return data

tc0 = TestClass('x', 0, 1)

d0 = tc0.to_data()
d1 = tc0.to_dunl_dict()

dunl = sfd.write_dunl_code(d1)
a    = sfd.read_dunl_code(dunl)
assert a == d0
 