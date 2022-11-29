from datetime import datetime

import addpath
import dunlin as dn
from dunlin.standardfile.dunl.writecode import *

###############################################################################
#Check if string needs quotes
###############################################################################
a = 'a'
assert needs_quotes(a) == False

a = "'a'"
assert needs_quotes(a) == False

a = "'1'"
assert needs_quotes(a) == False

a = '1'
assert needs_quotes(a) == True

a = '`a`'
assert needs_quotes(a) == True

a = '$'
assert needs_quotes(a) == True

a = '#'
assert needs_quotes(a) == True

a = '!'
assert needs_quotes(a) == True

###############################################################################
#Check write_primitive
###############################################################################
#Tests for strings and math
a = "'a'"
r = write_primitive(a)
assert r == a

a = "a"
r = write_primitive(a)
assert r == a

a = "\"'a'\""
r = write_primitive(a)
assert r == a

a = "'1+2'"
r = write_primitive(a)
assert r == a

a = '1+2'
r = write_primitive(a)
assert r == repr(a)

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
#Check write_list 
###############################################################################
#Single line no nesting
a = [0, '1', False, 'a', '"a"']
r = write_list(a)
assert r == "[0, '1', False, a, \"a\"]"

#Single line no nesting
a = [[0, 1], [0, 1], [0, 1]]
r = write_list(a)
assert r == '[[0, 1], [0, 1], [0, 1]]'

###############################################################################
#Check write_key
###############################################################################
a = ['a','b']
r = write_key(a)
assert r == '[a, b]'

###############################################################################
#Check write_directory
###############################################################################
a = ['a','b']
r = write_directory(a)
assert r == ';a;b'

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

a = {'a': 0, 'b': {1: 2, 3: 4}, 'c': 5}
r = write_dict(a, multiline_dict=True)
s = 'a : 0\nb : [\n\t1 : 2,\n\t3 : 4\n\t]\nc : 5'

#Uses tab (\t) instead of spaces to indent
assert r.strip() == s

a = {'a': 0, 'b': {1: 2, 3: {4: 5}, 6: 7}, 'c': 5}
r = write_dict(a, multiline_dict=True)
s = 'a : 0\nb : [\n\t1 : 2,\n\t3 : [\n\t\t4 : 5\n\t\t],\n\t6 : 7\n\t]\nc : 5'

#Uses tab (\t) instead of spaces to indent
assert r.strip() == s

#Nested with single-line list
a = {'a' : [0, 1, 2], 'b': [0, 1, 2]}
r = write_dict(a, multiline_dict=True)
s = 'a : [0, 1, 2]\nb : [0, 1, 2]'

#Uses tab (\t) instead of spaces to indent
assert r.strip() == s

#Nested with single-line list
a = {'a' : [0, 1, 2], 'b': [0, 1, [2, 3, 4]]}
r = write_dict(a, multiline_dict=True)
s = 'a : [0, 1, 2]\nb : [0, 1, [2, 3, 4]]'

#Uses tab (\t) instead of spaces to indent
assert r.strip() == s

###############################################################################
#Check write_dict with nested list
###############################################################################
a = [{'a': 1}, {'a': 1}]
r = write_list(a)
assert r.strip() == '[[a : 1], [a : 1]]'

a = [{'a': 1, 'b': 2}, {'a': 1}]
r = write_list(a)
assert r.strip() == '[[a : 1, b : 2], [a : 1]]'

a = {'a': [[0, 1], {'a': 1}]}
r = write_dict(a)
assert r.strip() == 'a : [[0, 1], [a : 1]]'


###############################################################################
#Check write_dunl_code
###############################################################################
#Write dunl_code
a = {'a': {0: 1}, 'b': {2: 3, 4: {5: 6}}, 'c': {7: 8, 9: 10}}
r = write_dunl_code(a)
r = r.strip().replace(' ', '')
assert r == ';a\n0:1\n\n;b\n2:3\n\n;b;4\n5:6\n\n;c\n7:8\n9:10'

a = {'a': {0: ['a', 'b', True]}, 'b': {2: 3, 4: {5: 6}}}
r = write_dunl_code(a)
r = r.strip().replace(' ', '')
assert r == ';a\n0:[a,b,True]\n\n;b\n2:3\n\n;b;4\n5:6'

a = {'a': {0: ['a', 'b', [True, False]]}, 'b': {2: 3, 4: {5: 6}}}
r = write_dunl_code(a)
r = r.strip().replace(' ', '')
assert r == ';a\n0:[a,b,[True,False]]\n\n;b\n2:3\n\n;b;4\n5:6'

class Custom:
    def to_dunl(self, x=True):
        if x:
            return '[b: 3, c: 4]'
        else:
            return '[\n\tb: 3,\n\tc: 4\n\t]'

a = {'a': Custom()}
r = write_dunl_code(a)
r = r.strip().replace(' ', '')
assert r == ';a\n[b:3,c:4]'

r = write_dunl_code(a, x=False)
r = r.strip().replace(' ', '')
assert r == ';a\n[\n\tb:3,\n\tc:4\n\t]'
# print(r)
