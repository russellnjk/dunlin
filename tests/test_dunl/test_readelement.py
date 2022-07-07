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
{{x}} : [`x` : [{x}]]
    $x   : {x}
    $x.x : 0, 1, 2, '"{4}"'
    $$x  : x0, x1
'''
interpolators = {'x' : 'c0'}

r = rel.read_element(element, interpolators)
assert r == {'x0': {'c0': [0, 1, 2, '{4}']}, 'x1': {'c0': [0, 1, 2, '{4}']}}

print(r)