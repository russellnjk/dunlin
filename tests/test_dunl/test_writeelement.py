from datetime  import datetime
from pyrfc3339 import generate

import addpath
from dunlin.standardfile.dunl.writeelement import (write_primitive, 
                                                   write_list, 
                                                   write_dict,
                                                   write_string_primitive
                                                   )

import dunlin.standardfile.dunl.writeelement as we

###############################################################################
#Check write_string_primitive
###############################################################################
a = '1' 
b = 'True'
c = 'c'
d = '"d"'
e = '\'"e"\''
f = '"'
g = '"g'
h = 'h!' 
i = '"f(x, y)"'
j = 'f(x, y)'

assert write_string_primitive(a) == repr(a)
assert write_string_primitive(b) == repr(repr(True))
assert write_string_primitive(c) == c
assert write_string_primitive(d) == d
assert write_string_primitive(e) == e
assert write_string_primitive(f) == repr(f)
assert write_string_primitive(g) == repr(g)
assert write_string_primitive(h) == repr(h)
assert write_string_primitive(i) == i
assert write_string_primitive(j) == repr(j)

# print(repr(write_string_primitive(f)))
###############################################################################
#Check write_primitive
###############################################################################
#Tests for strings
#Simple string
a = "a"
r = write_primitive(a)
assert r == a

#Tests for other primitives
a = 123
r = write_primitive(a)
assert r == repr(a)

a = True
r = write_primitive(a)
assert r == repr(a)

a = datetime.now()
r = write_primitive(a)
assert r == repr(generate(a, accept_naive=True, microseconds=True))

###############################################################################
#Check write_list without Dictionaries
###############################################################################
#No nesting
a = [0, '1', False, 'a', '"a"']
r = write_list(a)
assert r == "[0, '1', False, a, \"a\"]"

#Nesting but only with lists
a = [[0, 1], [0, 1], [0, 1]]
r = write_list(a)
assert r == '[[0, 1], [0, 1], [0, 1]]'

###############################################################################
#Check write_dict 
###############################################################################
#No nesting
a = {'a': 0, 'b': 1}
r = write_dict(a, multiline_dict=False)
assert r == 'a : 0, b : 1'

a = {'a': 0, 'b': 1}
r = write_dict(a, multiline_dict=True)
assert r == 'a : 0\nb : 1'

#Nested dict
a = {'a': 0, 'b': {1: 2, 3: 4}, 'c': 5}
r = write_dict(a, multiline_dict=False)
assert r == 'a : 0, b : [1 : 2, 3 : 4], c : 5'

a = {'a': 0, 'b': {1: 2, 3: {4: 5}, 6: 7}, 'c': 5}
r = write_dict(a, multiline_dict=False)
assert r == 'a : 0, b : [1 : 2, 3 : [4 : 5], 6 : 7], c : 5'

a = {'a': 0, 'b': {1: 2, 3: {4: 5}, 6: 7}, 'c': 5}
r = write_dict(a, multiline_dict=True)
s = 'a : 0\nb : [\n\t1 : 2\n\t3 : [\n\t\t4 : 5\n\t\t],\n\t6 : 7\n\t],\nc : 5'

###############################################################################
#Check write_dict with Nested Lists
###############################################################################
#Nested with list
a = {'a' : [0, 1, 2], 'b': [0, 1, 2]}
r = write_dict(a, multiline_dict=True)
s = 'a : [0, 1, 2]\nb : [0, 1, 2]'

#Nested with nested list
a = {'a' : [0, 1, 2], 'b': [0, 1, [2, 3, 4]]}
r = write_dict(a, multiline_dict=True)
s = 'a : [0, 1, 2]\nb : [0, 1, [2, 3, 4]]'

###############################################################################
#Check write_list with Nested Dicts
###############################################################################
a = [{'a': 1}, {'a': 1}]
r = write_list(a)
assert r == '[[a : 1], [a : 1]]'

a = [{'a': 1, 'b': 2}, {'a': 1}]
r = write_list(a)
assert r == '[[a : 1, b : 2], [a : 1]]'

a = {'a': [[0, 1], {'a': 1}]}
r = write_dict(a)
assert r == 'a : [[0, 1], [a : 1]]'
