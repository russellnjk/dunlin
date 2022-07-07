import addpath
import dunlin.utils as ut
from dunlin.comp.child import make_child_item
from dunlin.comp.flattenmodel import flatten_ode
from data import all_data

###############################################################################
#Test flatten
###############################################################################
#One level deep. No recursion.
parent_ref = 'm1'
flattened  = flatten_ode(all_data, parent_ref)

assert flattened['states'] == {'sm0.x0': {'c0': 3, 'c1': 2}, 
                               'sm0.x1': {'c0': 0, 'c1': 0}, 
                               'sm0.x2': {'c0': 0, 'c1': 0}, 
                               'sm0.x3': {'c0': 0, 'c1': 0}, 
                               'xx0'   : {'c0': 1, 'c1': 2}, 
                               'sm1.x1': {'c0': 0, 'c1': 0}, 
                               'xx2'   : {'c0': 0, 'c1': 0}, 
                               'sm1.x3': {'c0': 0, 'c1': 0}
                               }
assert flattened['parameters'] == {'sm0.p0': {'c0': 0.01, 'c1': 0.01}, 
                                   'sm0.p1': {'c0': 0.01, 'c1': 0.01}, 
                                   'sm0.p2': {'c0': 0.01, 'c1': 0.01}, 
                                   'sm1.p0': {'c0': 0.01, 'c1': 0.01}, 
                                   'sm1.p1': {'c0': 0.01, 'c1': 0.01}, 
                                   'sm1.p2': {'c0': 0.01, 'c1': 0.01}
                                   }
assert flattened['functions'] == {'sm0.f0': ['sm0.a', 'sm0.b', 'sm0.a*sm0.b'], 
                                  'sm1.f0': ['sm1.a', 'sm1.b', 'sm1.a*sm1.b']
                                  }
assert flattened['variables'] == {'sm0.v0': '-sm0.f0(sm0.p2, sm0.x2)', 
                                  'sm0.v1': 'sm0.f0(sm0.p2, sm0.x2)', 
                                  'sm1.v0': '-sm1.f0(sm1.p2, xx2)', 
                                  'sm1.v1': 'sm1.f0(sm1.p2, xx2)'
                                  }
assert flattened['rates'] == {'sm0.x2': 'sm0.v0', 
                              'sm0.x3': 'sm0.v1', 
                              'xx2'   : 'sm1.v0', 
                              'sm1.x3': 'sm1.v1'
                              }
assert flattened['reactions'] == {'sm0.r0': ['sm0.x0 -> sm0.x1', 'sm0.p0*sm0.x0'], 
                                  'sm0.r1': ['sm0.x1->', 'sm0.p1*sm0.x1'], 
                                  'sm1.r0': ['xx0 -> sm1.x1', 'sm1.p0*xx0'], 
                                  'sm1.r1': ['sm1.x1->', 'sm1.p1*sm1.x1']
                                  }
assert flattened['events'] == {'sm0.ev0': {'trigger': 'time==0', 
                                           'assign': ['sm0.x2 = 1', 
                                                      'sm0.x3 = 1'
                                                      ]
                                           }
                               }

#Zero levels deep. No recursion.
parent_ref = 'm0'
flattened = flatten_ode(all_data, parent_ref)
assert flattened['states'] == {'x0': {'c0': 1, 'c1': 2}, 
                                'x1': {'c0': 0, 'c1': 0}, 
                                'x2': {'c0': 0, 'c1': 0}, 
                                'x3': {'c0': 0, 'c1': 0}
                                }
assert flattened['parameters'] == {'p0': {'c0': 0.01, 'c1': 0.01}, 
                                    'p1': {'c0': 0.01, 'c1': 0.01}, 
                                    'p2': {'c0': 0.01, 'c1': 0.01}
                                    }
assert flattened['reactions'] == {'r0': ['x0 -> x1', 'p0*x0'], 
                                  'r1': ['x1->', 'p1*x1']
                                  }
assert flattened['functions'] == {'f0': ['a', 'b', 'a*b']
                                  }
assert flattened['variables'] == {'v0': '-f0(p2, x2)', 'v1': 'f0(p2, x2)'
                                  }
assert flattened['rates'] == {'x2': 'v0', 
                              'x3': 'v1'
                              }
assert flattened['events'] == {'ev0': {'trigger': 'time==0', 
                                        'assign': ['x2 = 1', 'x3 = 1'
                                                  ]
                                        }
                                }
assert flattened['sim_args'] == {'line_args': {'linestyle': ':'}}

#Two levels deep. One recursive call.
parent_ref = 'm2'
flattened  = flatten_ode(all_data, parent_ref)

assert flattened['states'] == {'sm0.x0': {'c0': 1, 'c1': 2}, 
                               'sm0.x1': {'c0': 0, 'c1': 0}, 
                               'sm0.x2': {'c0': 0, 'c1': 0}, 
                               'sm0.x3': {'c0': 0, 'c1': 0}, 
                               'sm1.sm0.x0': {'c0': 3, 'c1': 2}, 
                               'sm1.sm0.x1': {'c0': 0, 'c1': 0}, 
                               'sm1.sm0.x2': {'c0': 0, 'c1': 0}, 
                               'sm1.sm0.x3': {'c0': 0, 'c1': 0}, 
                               'sm1.xx0': {'c0': 1, 'c1': 2}, 
                               'sm1.sm1.x1': {'c0': 0, 'c1': 0}, 
                               'sm1.xx2': {'c0': 0, 'c1': 0}, 
                               'sm1.sm1.x3': {'c0': 0, 'c1': 0}, 
                               'sm0.sm0.x0': {'c0': 3, 'c1': 2}
                               }
assert flattened['parameters'] == {'sm0.p0': {'c0': 0.01, 'c1': 0.01}, 
                                   'sm0.p1': {'c0': 0.01, 'c1': 0.01}, 
                                   'sm0.p2': {'c0': 0.01, 'c1': 0.01}, 
                                   'sm1.sm0.p0': {'c0': 0.01, 'c1': 0.01}, 
                                   'sm1.sm0.p1': {'c0': 0.01, 'c1': 0.01}, 
                                   'sm1.sm0.p2': {'c0': 0.01, 'c1': 0.01}, 
                                   'sm1.sm1.p0': {'c0': 0.01, 'c1': 0.01}, 
                                   'sm1.sm1.p1': {'c0': 0.01, 'c1': 0.01}, 
                                   'sm1.sm1.p2': {'c0': 0.01, 'c1': 0.01}
                                   }
assert flattened['functions'] == {'sm0.f0': ['sm0.a', 'sm0.b', 'sm0.a*sm0.b'], 
                                  'sm1.sm0.f0': ['sm1.sm0.a', 'sm1.sm0.b', 'sm1.sm0.a*sm1.sm0.b'], 
                                  'sm1.sm1.f0': ['sm1.sm1.a', 'sm1.sm1.b', 'sm1.sm1.a*sm1.sm1.b']
                                  }
assert flattened['variables'] == {'sm0.v0': '-sm0.f0(sm0.p2, sm0.x2)', 
                                  'sm0.v1': 'sm0.f0(sm0.p2, sm0.x2)', 
                                  'sm1.sm0.v0': '-sm1.sm0.f0(sm1.sm0.p2, sm1.sm0.x2)', 
                                  'sm1.sm0.v1': 'sm1.sm0.f0(sm1.sm0.p2, sm1.sm0.x2)', 
                                  'sm1.sm1.v0': '-sm1.sm1.f0(sm1.sm1.p2, sm1.xx2)', 
                                  'sm1.sm1.v1': 'sm1.sm1.f0(sm1.sm1.p2, sm1.xx2)'
                                  }
assert flattened['rates'] == {'sm0.x2': 'sm0.v0', 
                              'sm0.x3': 'sm0.v1', 
                              'sm1.sm0.x2': 'sm1.sm0.v0', 
                              'sm1.sm0.x3': 'sm1.sm0.v1', 
                              'sm1.xx2': 'sm1.sm1.v0', 
                              'sm1.sm1.x3': 'sm1.sm1.v1'
                              }
assert flattened['reactions'] == {'sm0.r0': ['sm0.x0 -> sm0.x1', 'sm0.p0*sm0.x0'], 
                                  'sm0.r1': ['sm0.x1->', 'sm0.p1*sm0.x1'], 
                                  'sm1.sm0.r0': ['sm1.sm0.x0 -> sm1.sm0.x1', 'sm1.sm0.p0*sm1.sm0.x0'], 
                                  'sm1.sm0.r1': ['sm1.sm0.x1->', 'sm1.sm0.p1*sm1.sm0.x1'], 
                                  'sm1.sm1.r0': ['sm1.xx0 -> sm1.sm1.x1', 'sm1.sm1.p0*sm1.xx0'], 
                                  'sm1.sm1.r1': ['sm1.sm1.x1->', 'sm1.sm1.p1*sm1.sm1.x1']
                                  }
assert flattened['events'] == {'ev0': {'trigger': 'time==0', 
                                       'assign': ['sm0.x2 = 1', 'sm0.x3 = 1']
                                       }
                               }

#Circular hierarchy
parent_ref = 'm3'
try:
    flattened = flatten_ode(all_data, parent_ref)
except:
    assert True
else:
    assert False

parent_ref = 'm4'
try:
    flattened = flatten_ode(all_data, parent_ref)
except:
    assert True
else:
    assert False
    