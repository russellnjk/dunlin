import addpath
import dunlin as dn
from dunlin.spatial.grid.bidict import One2One, One2Many

a = One2One('x', 'y')
b = a.inverse

a[1] = 10
a[2] = 11
assert a[1]  == 10
assert b[10] == 1
assert a[2]  == 11
assert b[11] == 2

a[3] = 10
assert a[3]  == 10
assert b[10] == 3
    
a = One2Many('x', 'y')
b = a.inverse

a[1] = 10
a[2] = 10

assert a[1] == 10
assert 1 in b[10]
assert a[2] == 10
assert 2 in b[10]

a[2] = 9

assert a[2] == 9
assert 2 in b[9]
assert 2 not in b[10]

