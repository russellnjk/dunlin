import numpy  as np
import pandas as pd
from datetime import datetime

import addpath
import dunlin as dn
from dunlin.standardfile.dunl.writefile import *
from dunlin.standardfile.dunl.readdunl  import read_dunl_code

###############################################################################
#Frontend Conversion of Data to dunl code
###############################################################################
a = {'a' : {0: 1}, 'b': {0: 1}}
r = write_dunl_code(a)
assert read_dunl_code(r) == a

a = {'a' : {0: 1}, 'b': {0: [1, 2]}, 'c': {0: 1}}
r = write_dunl_code(a)
assert read_dunl_code(r) == a

a = {'a' : {0: {'x' : [0, 1], 'y': [0, 1]}}, 'b': {0: [1, 2]}, 'c': {0: 1}}
r = write_dunl_code(a)
assert read_dunl_code(r) == a


a = {'a' : {0: {'b': {'c': {'d' : [0, 1, 2]}
                      }
                }
            }
     }
r = write_dunl_code(a, max_dir=1)
assert read_dunl_code(r) == a

r = write_dunl_code(a, max_dir=1, multiline_dict=False)
assert read_dunl_code(r) == a

# print(r)

###############################################################################
#File Functions
###############################################################################
filename = 'temp.dunl'
r = write_dunl_file(a, filename=filename)
assert read_dunl_code(r) == a

with open(filename) as file:
    c = file.read()
print(c)
print(r)


