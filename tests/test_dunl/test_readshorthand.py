import addpath
import dunlin as dn
import dunlin.standardfile.dunl.readshorthand as rsh

import dunlin.standardfile.dunl.delim as dm

from dunlin.standardfile.dunl.readshorthand import (format_string, 
                                                    zip_substitute,
                                                    substitute_horizontals,
                                                    substitute_verticals,
                                                    substitute,
                                                    string2chunks,
                                                    read_shorthand
                                                    )


a = '"{a}", {a}'
r = format_string(a, a=2)
assert r == '"{a}", 2'

a = '["{a}", {a}]'
r = format_string(a, a=2)
assert r == '["{a}", 2]'

a = '[{a} : "{a}"]'
r = format_string(a, a=2)
assert r == '[2 : "{a}"]'

a = '"{a}", {a}, {{b}}'
r = format_string(a, a=2)
assert r == '"{a}", 2, {b}'

a = '"{a}", {a}, {{b}}, {{ {c} }}'
try:
    r = format_string(a, a=2)
except Exception:
    assert True
else:
    assert False

#Set up horizontal
horizontal          = rsh.Horizontal({'key'   : ['0', '1'],
                                      'value' : ['2', '3']
                                      }
                                      )
horizontal.join     = ', '
horizontal.template = '[{key}: {value}]' 

#Test single horizontal sub
r = zip_substitute(horizontal.template, horizontal)
r = horizontal.join.join(r)
assert r == '[0: 2], [1: 3]'

#Test iterative horizontal sub
template    = '{{x}} = {a}'
horizontals = {'a': horizontal}
r = substitute_horizontals(template, horizontals)
assert r == '{x} = [0: 2], [1: 3]'

#Test verticals
template  = '{x} = [0: 2], [1: 3]'
verticals = {'x': ['x0', 'x1']}
r = substitute_verticals(template, verticals)
assert r == ['x0 = [0: 2], [1: 3]', 'x1 = [0: 2], [1: 3]']

template  = '{{x}} = [0: 2], [1: 3]'
verticals = {'x': ['x0', 'x1']}
try:
    r = substitute_verticals(template, verticals)
except:
    assert True
else:
    assert False

#Test substitution
template    = '{{x}} = {a}'
horizontals = {'a': horizontal}
verticals = {'x': ['x0', 'x1']}
substitute(template, horizontals, verticals)
assert r == ['x0 = [0: 2], [1: 3]', 'x1 = [0: 2], [1: 3]']

#Test with quotes and curly braces
horizontal          = rsh.Horizontal({'key'   : ['0', '1'],
                                      }
                                      )
horizontal.join     = ', '
horizontal.template = '[{key}: "{s}"]' 

template    = '{{x}} = {a}'
horizontals = {'a': horizontal}
verticals = {'x': ['x0', 'x1']}
r = substitute(template, horizontals, verticals)
assert r == ['x0 = [0: "{s}"], [1: "{s}"]', 
             'x1 = [0: "{s}"], [1: "{s}"]'
             ]

#Test string2chunks
element = '[{a}] : "{s}" $a : {i} $a.i : 0, 1, 2 '
r = string2chunks(element)
template, shorthands = r
assert template   == '[{a}] : "{s}" '
assert shorthands == [['$', 'a ', ' {i} '], 
                      ['$', 'a.i ', ' 0, 1, 2 ']
                      ]

#Read shorthands
element = '[{a}] : "{s}" $a : {i} $a.i : 0, 1, 2 '
r = read_shorthand(element)

# print(r)
# print(repr(r))

