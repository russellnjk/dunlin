import addpath
import dunlin as dn
import dunlin.standardfile.dunl.parsepath as pp

#Test read_key
a = 'a'
r = pp.read_key(a)
assert r == 'a'

a = '0'
r = pp.read_key(a)
assert r == 0

a = '"0"'
r = pp.read_key(a)
assert r == "0"

a = 'True'
r = pp.read_key(a)
assert r == True

a = '[0, 1]'
r = pp.read_key(a)
assert r == (0, 1)

#Test split_path
a = ';a;bc;de'
r = pp.split_path(a)
assert r == ['a', 'bc', 'de']

a = ';;; ;'
r = pp.split_path(a)
assert r == ['', '', '', '']

a = ';[0, 1];True;(1+2)'
r = pp.split_path(a)
assert r == [(0, 1), True, 3]

a = ';0, 1;True;(1+2)'
try:
    r = pp.split_path(a)
except Exception:
    assert True
else:
    assert False

a = ';0;" "'
try:
    r = pp.split_path(a)
except Exception:
    assert True
else:
    assert False
    
#Test replace relative path
a = ['', '', 0]
b = ['a', 'b']
r = pp.replace_relative_path(a, b)
assert r == ['a', 'b', 0]

a = ['', '', 0]
b = ['a', 'b', 'c', 'd']
r = pp.replace_relative_path(a, b)
assert r == ['a', 'b', 0]

a = ['', '', '', 0]
b = ['a', 'b']
try:
    r = pp.replace_relative_path(a, b)
except Exception:
    assert True
else:
    assert False

#Test recurse
d = {}
r = pp.recurse(d, ['a', 'b'])

assert r == {}
assert d == {'a': {'b': {}}}
assert r is d['a']['b']

d = {'a' : {0 : 1}}
r = pp.recurse(d, ['a', 'b'])

assert r == {}
assert d == {'a': {0: 1, 'b': {}}}
assert r is d['a']['b']

#Test go_to
d = {}
c = []
dst, c_ = pp.go_to(d, ['a', 'b'], c)
assert c_ == ['a', 'b']
assert d  == {'a': {'b': {}}}
assert dst is d['a']['b']
assert c_ is c

dst, c_ = pp.go_to(d, ['x'], c)
assert c_ == ['x']
assert d  == {'a': {'b': {}}, 'x': {}}
assert dst is d['x']
assert c_ is c

print(r)



