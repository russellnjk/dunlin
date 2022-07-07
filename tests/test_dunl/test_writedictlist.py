from datetime import datetime

import addpath
import dunlin as dn
from dunlin.standardfile.dunl.writedictlist import *

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
s = '''
a : 0
b : [
	1 : 2,
	3 : 4
	],
c : 5
'''
#Uses tab (\t) instead of spaces to indent
assert r == s.strip()

a = {'a': 0, 'b': {1: 2, 3: {4: 5}, 6: 7}, 'c': 5}
r = write_dict(a, multiline_dict=True)
s = '''
a : 0
b : [
	1 : 2,
	3 : [
		4 : 5
		],
	6 : 7
	],
c : 5
'''
#Uses tab (\t) instead of spaces to indent
assert r == s.strip()

#Nested with single-line list
a = {'a' : [0, 1, 2], 'b': [0, 1, 2]}
r = write_dict(a, multiline_dict=True)
s = '''
a : [0, 1, 2]
b : [0, 1, 2]
'''
#Uses tab (\t) instead of spaces to indent
assert r == s.strip()

#Nested with single-line list
a = {'a' : [0, 1, 2], 'b': [0, 1, [2, 3, 4]]}
r = write_dict(a, multiline_dict=True)
s = '''
a : [0, 1, 2]
b : [0, 1, [2, 3, 4]]
'''
#Uses tab (\t) instead of spaces to indent
assert r == s.strip()

###############################################################################
#Check write_dict with nested list
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
