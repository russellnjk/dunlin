import addpath
import dunlin.utils as ut
# from dunlin.comp.child import wrap_merge, make_child_item
# from data              import all_data
from dunlin.comp.child import make_child_item

r = make_child_item('a+bb+c.d+e1', 'm0', {'e1': 'f2'})
assert r == 'm0.a+m0.bb+m0.c.d+f2'

r = make_child_item('a+bb+c.d+e1.a', 'm0', {'e1.a': 'f2'})
assert r == 'm0.a+m0.bb+m0.c.d+f2'

print(r)
