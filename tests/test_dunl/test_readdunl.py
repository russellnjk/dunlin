import addpath
import dunlin as dn
import dunlin.standardfile.dunl.readdunl as rdn

'''
`a` 0

;0
1 : [2 : 3,
     4 : 5
     ]

#Comment

'''

###############################################################################
#Test Comment Removal
###############################################################################
line = 'abc "#abc" #abc'
r    = rdn.remove_comments(line)
# print(r)
assert r == 'abc "#abc"'

###############################################################################
##Test read_chunk
###############################################################################
'''
Updates dct, curr_lst and interpolators during each call.
Returns the new curr_dct

Updates curr_dct after every chunk outside the function.
'''
dct           = {}
curr_lst      = []
curr_dct      = dct
interpolators = {}

#Interpolator
chunk = '`x` 0'

curr_dct = rdn.read_chunk(dct, curr_lst, curr_dct, interpolators, chunk)
assert interpolators == {'x': '0'}
assert dct           == {} 
assert curr_lst      == []
assert curr_dct is dct

#Directory
chunk = ';a;b'

curr_dct = rdn.read_chunk(dct, curr_lst, curr_dct, interpolators, chunk)
assert interpolators == {'x': '0'}
assert dct      == {'a': {'b': {}}}
assert curr_lst == ['a', 'b']
assert curr_dct is dct['a']['b']

#Element
chunk = '''
{x} : [`x`: [!range, `x`, 1, 0.5!]]
    $x : j, k
'''

curr_dct = rdn.read_chunk(dct, curr_lst, curr_dct, interpolators, chunk)
assert interpolators == {'x': '0'}
assert dct['a']['b'] == {'j': {0: [0.0, 0.5]}, 'k': {0: [0.0, 0.5]}}
assert curr_dct is dct['a']['b']

#Ambiguous/erroneous chunk
chunk = ';a;b : 5'
try:
    r = rdn.read_chunk(dct, curr_lst, curr_dct, interpolators, chunk)
except:
    assert True
else:
    assert False

chunk = '`y 4'
try:
    r = rdn.read_chunk(dct, curr_lst, curr_dct, interpolators, chunk)
except:
    assert True
else:
    assert False

chunk = '`y'
try:
    r = rdn.read_chunk(dct, curr_lst, curr_dct, interpolators, chunk)
except:
    assert True
else:
    assert False


#Test read_lines
code = '''
`x` 0

;a;b
{x} : [`x` : [0, 1, 2]]
    $x  : x0, x1

# haha
# ;c #hehe
# y : 3

'''
lines = code.split('\n')

r = rdn.read_lines(lines)
assert r == {'a': {'b': {'x0': {0: [0, 1, 2]}, 
                          'x1': {0: [0, 1, 2]}
                          }
                    }
              }

# print(r)

#Test on file
r  = rdn.read_dunl_file('example_data.dunl')

assert r['a']['aa'] == {'aa0': 0,
                        'aa1': [1, 2],
                        'aa2': [1, 1],
                        'aa3': {'x': 'xx', 'y': 'yy', 'z': 'zz'},
                        'aa4': {'x': 'xx', 'y': 'yy', 'z': 'zz'}
                        }

assert r['b']['horizontal'] == {'x0': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                                'x1': [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                                'x2': ['a', 'c', 'e', 'g', 'i', 'k', 'm', 'o', 'q'],
                                'x3': ['a', 'c', 'e', 'g', 'i', 'k', 'm', 'o'],
                                'x4': ['constant_abc',
                                       'constant_def',
                                       'constant_hij',
                                       'constant_klm',
                                       'constant_nop',
                                       'constant_qrs',
                                       'constant_tuv',
                                       'constant_wxy'],
                                'x5': ['constant_abc',
                                       'constant_def',
                                       'constant_hij',
                                       'constant_klm',
                                       'constant_nop',
                                       'constant_qrs',
                                       'constant_tuv',
                                       'constant_wxy'],
                                'x6': 'constant_abc + constant_def + constant_hij + constant_klm + constant_nop + constant_qrs + constant_tuv + constant_wxy',
                                'x7': 'constant_abc + constant_def + constant_hij + constant_klm + constant_nop + constant_qrs + constant_tuv + constant_wxy'
                                }

assert r['b']['vertical']['no_shorthands'] == {'x': {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4},
                                               'y': {'a': 0, 'b': 1, 'c': 3, 'd': 3, 'e': 4},
                                               'z': {'a': 0, 'b': 1, 'c': 4, 'd': 3, 'e': 4}
                                               }

assert r['b']['vertical']['with_shorthands'] == r['b']['vertical']['no_shorthands']

assert r['b']['interpolation'] == {'q': {'a': 0, 'b': 1, 'c': 2, 'd': 3,  'e': 4},
                                   'r': {'a': 0, 'b': 1, 'c': 2, 'd': 4,  'e': 4},
                                   's': {'a': 0, 'b': 1, 'c': 2, 'd': 5,  'e': 4},
                                   't': {'a': 0, 'b': 1, 'c': 2, 'd': 6,  'e': 4},
                                   'u': {'a': 0, 'b': 1, 'c': 2, 'd': 7,  'e': 4},
                                   'v': {'a': 0, 'b': 1, 'c': 2, 'd': 8,  'e': 4},
                                   'w': {'a': 0, 'b': 1, 'c': 2, 'd': 9,  'e': 4},
                                   'x': {'a': 0, 'b': 1, 'c': 2, 'd': 10, 'e': 4},
                                   'y': {'a': 0, 'b': 1, 'c': 2, 'd': 11, 'e': 4},
                                   'z': {'a': 0, 'b': 1, 'c': 2, 'd': 12, 'e': 4},
                                   'some_quantity': '( q + r + s + t + u + v + w + x + y + z )/2'
                                   }
