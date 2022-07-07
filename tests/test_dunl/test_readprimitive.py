from datetime import datetime

import addpath
import dunlin as dn
import dunlin.standardfile.dunl.readprimitive as rpr

#Basic cases
a = 'a'
b = '0'
c = '1.5'
d = 'a.b'
e = 'a + b'
f = '1 + 2'
g = '1979-05-27T07:32:00Z'
h = 'True'
i = '!linspace(0, 10, 11)'

r = rpr.read_primitive(a)
assert r == 'a'

r = rpr.read_primitive(b)
assert r == 0

r = rpr.read_primitive(c)
assert r == 1.5

r = rpr.read_primitive(d)
assert r == 'a.b'

r = rpr.read_primitive(e)
assert r == 'a + b'

r = rpr.read_primitive(f)
assert r == 3

r = rpr.read_primitive(g)
assert type(r) == datetime

r = rpr.read_primitive(h)
assert r is True

r = rpr.read_primitive(i)
assert list(r) == list(range(11))

#Trick cases
a = 'a.b.3'
b = '1.a'
c = '(a)'
d = '(a, b)'

r = rpr.read_primitive(a)
assert r == 'a.b.3'

r = rpr.read_primitive(b)
assert r == '1.a'

r = rpr.read_primitive(c)
assert r == '(a)'

r = rpr.read_primitive(d)
assert r == '(a, b)'
