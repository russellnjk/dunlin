import addpath
import dunlin as dn
from dunlin.standardfile.dunl.readshorthand import (read_shorthand,
                                                    read_horizontal,
                                                    split_interpolated,
                                                    read_vertical
                                                    )

###############################################################################
#Test Horizontal Shorthands
###############################################################################
#Assume interpolation is complete
#0, 0.5, 1, 1.5, 2
interpolated = '!range, 0, 2, 0.5!'
expanded     = read_horizontal(interpolated)
# print(expanded)
assert expanded == '0.000000, 0.500000, 1.000000, 1.500000'

#k_a_0, k_a_1, k_b_0, k_b_1
interpolated = '!comma, k_{}_{}, a, 0, a, 1, b, 0!'
expanded     = read_horizontal(interpolated)
# print(expanded)
assert expanded == 'k_a_0, k_a_1, k_b_0'

#Test with faulty input
interpolated = '!comma, k_{}_{}, a, 0, a, 1, b, 0,, !'
try:
    expanded     = read_horizontal(interpolated)
except Exception:
    assert True
else:
    assert False

#k_a_0, k_a_1, k_b_0, k_b_1
interpolated = '!plus, k_{}_{}, a, 0, a, 1, b, 0!'
expanded     = read_horizontal(interpolated)
# print(expanded)
assert expanded == 'k_a_0 + k_a_1 + k_b_0'

#k_a_0, k_a_1, k_b_0, k_b_1
interpolated = '!plus, zip, k_{}_{}, a, a, b, 0, 1, 0!'
expanded     = read_horizontal(interpolated)
# print(expanded)
assert expanded == 'k_a_0 + k_a_1 + k_b_0'

###############################################################################
#Test Vertical Shorthands
###############################################################################
interpolated = '''
{mykey} : [c0: {myvalue}]
    $mykey   : a, b, c
    $myvalue : 0, 1, 2
'''

template, shorthands = split_interpolated(interpolated)
# print(template)
# print(shorthands)

assert template   == '{mykey} : [c0: {myvalue}]' 
assert shorthands == {'mykey': ['a', 'b', 'c'], 'myvalue': ['0', '1', '2']}

strings = read_vertical(interpolated)
print(strings)
assert strings == ['a : [c0: 0]', 'b : [c0: 1]', 'c : [c0: 2]']

###############################################################################
#Test Vertical Shorthands
###############################################################################
interpolated = '''
{mykey} : [!range, 0, 2, 0.5!]
    $mykey   : a, b, c
'''
strings = read_shorthand(interpolated)
print(strings)

assert strings == ['a : [0.000000, 0.500000, 1.000000, 1.500000]', 
                   'b : [0.000000, 0.500000, 1.000000, 1.500000]', 
                   'c : [0.000000, 0.500000, 1.000000, 1.500000]'
                   ]

