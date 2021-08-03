import numpy as np
from requests import *

states = {'x0' : [1, 1],
          'x1' : [0, 0],
          'x2' : [1, 1],
          'x3' : [0, 0]
          }

params = {'p0' : [0.1],
          'p1' : [0.1],
          'p2' : [0.1],
          'p3' : [0.1]
          }

rxns = {'r0': ['x0 >   ', 'p0'],
        'r1': ['   > x1', 'p1']
        }
