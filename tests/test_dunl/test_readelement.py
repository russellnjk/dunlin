import addpath
import dunlin as dn
import dunlin.standardfile.dunl.readelement as rel

#Test interpolation
element = 'a b `x`'
interpolators = {'x' : '"c"'}

r = rel.interpolate(element, interpolators)
assert r == 'a b "c"'

element = 'a b `x` c'
interpolators = {'x' : '"c"'}

r = rel.interpolate(element, interpolators)
assert r == 'a b "c" c'

element = 'a b `y` c'
interpolators = {'x' : '"c"'}

try:
    r = rel.interpolate(element, interpolators)
except:
    assert True
else:
    assert False

# print(repr(r))

#Test read_element
element = '''
{x} : [`x` : [!range, 0, 1, 2!]]
    $x  : x0, x1
'''
interpolators = {'x' : 'c0'}

r = rel.read_element(element, interpolators)
assert r == {'x0': {'c0': [0.0]}, 'x1': {'c0': [0.0]}}

print(r)

element = '''
{x} : [c0: [!range, 0, 1, 0.5!]]
    $x : j, k
'''

element = '''
{x} : [c0: [!range, 0, 1, 0.5!]]
    $x: j, k 
'''

interpolators = {'t' : '0'}

r = rel.read_element(element, interpolators)
print(r)
