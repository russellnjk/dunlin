import numpy as np
from numba import njit

n = 10000
x = np.arange(n)
a = np.ones(n)
s = f'''
@njit
def f(x):
    a = np.array({[1]*n})
    
    b = x + a
    return b

@njit
def g(x, a):
    b = x + a
    return b
'''

print('Exec code')
scope = {}
exec(s, {'np': np, 'njit': njit}, scope)

print('Compiling g')
g = scope['g']
g(x, a)

print('Compiling f')
f = scope['f']
f(x)


