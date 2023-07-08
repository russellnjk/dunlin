import addpath
import readstring as rst

###############################################################################
#Test Parsing Individual Values
###############################################################################
#Test read_value
'''
Parses values. Two cases are possible:
    1. x is dunl builtin function call.
    2. x is a primitive.
'''
a = 'a'
r = rst.read_value(a)
assert r == 'a'

a = '(a, b)'
r = rst.read_value(a)
assert r == '(a, b)'

a = 'a'
r = rst.read_value(a)
assert r == 'a'

'''
IMPORTANT: Parsing builtin functions will be left for later
'''

#Test read_key
'''
Parses keys. Three cases are possible:
    1. x is a list of primitives.
    2. x is a dunl builtin function call.
    3. x is a primitive
Because Python does not allow lists to used as keys, lists must be converted 
to tuples.
'''
a = 'a'
r = rst.read_key(a)
assert r == 'a'

a = ['a', 'b']
r = rst.read_key(a)
assert r == ('a', 'b')
assert type(r) == tuple

try:
    a = '" "'
    r = rst.read_key(a)
except:
    assert True
else:
    assert False

###############################################################################
#Test Parsing Flattened Data
###############################################################################
#Test read_list
a = ['a', 'b', 'c']
r = rst.read_list(a)
assert r == ['a', 'b', 'c']

a = ['a', 'b', ['c', 'd']]
r = rst.read_list(a)
assert r == ['a', 'b', ['c', 'd']]

a = ['a', 'b', ['c', 'd']*2]
r = rst.read_list(a)
assert r == ['a', 'b', ['c', 'd', 'c', 'd']]

#Test read_dict
a = ['a', ':', 'b', 'c', ':', 'd']
r = rst.read_dict(a)
assert r == {'a': 'b', 'c': 'd'}

a = ['a', ':',  'b', 'c', ':', [0, 1, 2]]
r = rst.read_dict(a)
assert r == {'a': 'b', 'c': [0, 1, 2]}

a = ['a', ':',  'b', 'c', ':', {'a': 'b'}]
r = rst.read_dict(a)
assert r == {'a': 'b', 'c': {'a': 'b'}}

a = ['a', ':',  ['b'], '*2', 'c', ':', {'a': 'b'}]
r = rst.read_dict(a)
assert r == {'a': ['b', 'b'], 'c': {'a': 'b'}}

#Test flat
'''
read_flat is a function that determines if a is a list or dict and calls the 
appropriate list/dict reader
'''
a = ['a', 'b', 'c']
r = rst.read_flat(a)
assert r == ['a', 'b', 'c']

a = ['a', 'b', ['c', 'd']]
r = rst.read_flat(a)
assert r == ['a', 'b', ['c', 'd']]

#Test read_dict
a = ['a', ':',  'b', 'c', ':', {'a': 'b'}]
r = rst.read_flat(a)
assert r == {'a': 'b', 'c': {'a': 'b'}}

a = ['a', ':',  ['b'], '*2', 'c', ':', {'a': 'b'}]
r = rst.read_flat(a)
assert r == {'a': ['b', 'b'], 'c': {'a': 'b'}}

###############################################################################
#Test _read_string
###############################################################################
'''
The goal of this function is to break the string into a nested list of tokens.
A list containing semicolon `:` is a corresponds to `dict`. A list not 
containing one is a `list`.

Pseudocode for algorithm

def _read_string(string, _flat_reader=lambda x: x):
    string = preprocess_string(string)
    i0     = 0
    nested = []
    curr   = []
    quotes = []
    
    for i, char in enumerate(string):
        if char == ',' and not quotes:
            append string[i0:i] to curr
            
            i0 = i + 1
            
        elif char == ':' and not quotes:
            append string[i0:i] to curr 
            append string[i:i+1] to curr
            
            i0 = i + 1
        
        elif char == '[' and not quotes:
            append string[i0:i] to curr
            
            nested.append(curr)
            curr = []
            
            i0 = i + 1
        
        elif char == ']' and not quotes:
            if not nested:
                raise Error()
                
            append string[i0:i] to curr
            
            parsed = _flat_reader(curr)
            curr   = nested.pop()
            curr.append(parsed)
            
            i0 = i + 1
        
        elif char is_quote:
            if not quotes:
                quotes.append(char)
            elif char == quotes[-1]:
                quotes.pop()
            else:
                quotes.append(char)
        
        else:
            continue
    
    append string[i0:i] to curr

    result = _flat_reader(curr)
    
    if len(nested) or quotes:
        raise Error()
    
    return result

'''
a = '[a, b] : [c : d]'
r = rst._read_string(a)
assert r == [['a', 'b'], ':', ['c', ':', 'd']]

a = 'a, b'
r = rst._read_string(a)
assert r == ['a', 'b']

a = 'a : b, c : d'
r = rst._read_string(a)
assert r == ['a', ':', 'b', 'c', ':', 'd']

#Test _read_string with quotes
a = '"a : b", "c : d"'
r = rst._read_string(a)
assert r == ['"a : b"', '"c : d"']

a = '":" : ":"'
r = rst._read_string(a)
assert r == ['":"', ':', '":"']

a = '"," , ","'
r = rst._read_string(a)
assert r == ['","', '","']

a = '[a, b]*2'
r = rst._read_string(a)
assert r == [['a', 'b'], '*2']

a = '0 : [a, b]*2, 1: 2'
r = rst._read_string(a)
assert r == ['0', ':', ['a', 'b'], '*2', '1', ':', '2']

a = 'f : [x, y, x + y]'
r = rst._read_string(a)
assert r == ['f', ':', ['x', 'y', 'x + y']]

###############################################################################
#Syntax Errors
###############################################################################
#Illegal starts
a = ', a'
try:
    r = rst._read_string(a)
except:
    assert True
else:
    assert False

a = ': a'
try:
    r = rst._read_string(a)
except:
    assert True
else:
    assert False

#Illegal delimiters after open bracket
a = 'a : [,0]'
try:
    r = rst._read_string(a)
except:
    assert True
else:
    assert False

a = 'a : [:0]'
try:
    r = rst._read_string(a)
except:
    assert True
else:
    assert False

#Illegal consecutive delimiters
a = 'a : [,,0]'
try:
    r = rst._read_string(a)
except:
    assert True
else:
    assert False

a = 'a :, [0]'
try:
    r = rst._read_string(a)
except:
    assert True
else:
    assert False

a = 'a , : [0]'
try:
    r = rst._read_string(a)
except:
    assert True
else:
    assert False

a = 'a :: [0]'
try:
    r = rst._read_string(a)
except:
    assert True
else:
    assert False

#Values bef open
a = 'a : 0 [0]'
try:
    r = rst._read_string(a)
except:
    assert True
else:
    assert False

#Values aft close but not mul
a = ' a : [0,] 2'
try:
    r = rst._read_string(a)
except:
    assert True
else:
    assert False

#Illegal trailing delimiter
a = ' a : [0],,'
try:
    r = rst._read_string(a)
except:
    assert True
else:
    assert False

#This is ALLOWED
a = ' a : [0],'
r = rst._read_string(a)
assert r == ['a', ':', ['0']]

#Nesting error
a = 'a : [0]], b:1'
try:
    r = rst._read_string(a)
except:
    assert True
else:
    assert False

a = 'a : [0], b: [1'
try:
    r = rst._read_string(a)
except:
    assert True
else:
    assert False
    
###############################################################################
#Combine read_flat with _read_string using read_string
###############################################################################
a = '"a : b", "c : d"'
r = rst.read_string(a)
assert r == {0: 'a : b', 1: 'c : d'}

a = 'f : [x, y, x + y]'
r = rst.read_string(a)
assert r == {'f' :['x', 'y', 'x + y']}

#Enforce dict
a = 'a, b, c'
r = rst.read_string(a)
assert r == {0: 'a', 1: 'b', 2: 'c'}

#Allow list at top level
a = 'a, b, c'
r = rst.read_string(a, enforce_dict=False)
assert r == ['a', 'b', 'c']

#Test parsing builtins
a = '!linspace, 0, 10, 11!'
r = rst.read_value(a)
assert r == list(range(11))

a = '!linspace, 0, 10, 11!'
r = rst.read_key(a)
assert r == tuple(range(11))
assert type(r) == tuple

a = '[0, 1] : !linspace, 0, 10, 11!'
r = rst._read_string(a)
assert r == [['0', '1'], ':', '!linspace, 0, 10, 11!']

a = '[0, 1] : !linspace, 0, 10, 11!'
r = rst.read_string(a)
assert r == {(0, 1) : list(range(11))}

try:
    a = 'f(x, y) : x + y'
    r = rst.read_string(a)
except:
    assert True
else:
    assert False
    
print(r)
