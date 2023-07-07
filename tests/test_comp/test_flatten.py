import addpath
import dunlin.utils as ut
from dunlin.comp.child import make_child_item
from dunlin.comp.flatten import dot, rename_delete, flatten
from data import all_data

###############################################################################
#Test dot
###############################################################################
child_name = 'sm1'
rename        = all_data['m1']['submodels'][child_name]['rename']
states        = all_data['m0']['states']

#Test with recurse set to [True, False, False]
#This is for data like states which only require renaming at the top level
new_value = dot(states, child_name, rename, set(), recurse=[True, False, False])
# print(new_value)

assert new_value == {'xx0'   : {'c0': 1, 'c1': 2}, 
                     'sm1.x1': {'c0': 0, 'c1': 0}, 
                     'xx2'   : {'c0': 0, 'c1': 0}, 
                     'sm1.x3': {'c0': 0, 'c1': 0}
                     }

#This is for data like reactions which require renaming at deeper levels
reactions = all_data['m0']['reactions']
new_value = dot(reactions, child_name, rename, set(), recurse=[True, True, True])
# print(new_value)

assert new_value == {'sm1.r0': ['xx0 -> sm1.x1', 'sm1.p0*xx0'], 
                     'sm1.r1': ['sm1.x1->', 'sm1.p1*sm1.x1']
                     }

events    = all_data['m0']['events']
new_value = dot(events, child_name, rename, set(), recurse=[True, False, True])
# print(new_value)

assert new_value == {'sm1.ev0': {'trigger': 'sm1.x1 < 0.01', 
                                 'assign': ['xx2 = 1', 
                                            'sm1.x3 = 1'
                                            ]
                                 }
                     }

#With deletion
events    = all_data['m0']['events']
delete    = all_data['m1']['submodels'][child_name]['delete']
new_value = dot(events, child_name, rename, delete, recurse=[True, False, True])
# print(new_value)

assert new_value == {}

###############################################################################
#Test delete_rename
###############################################################################
child_name = 'sm1'
delete        = set(all_data['m1']['submodels'][child_name]['delete'])
rename        = all_data['m1']['submodels'][child_name]['rename']

required_fields = {'states'     : [True, False, False],
                   'parameters' : [True, False, False],
                   'functions'  : [True, False],
                   'variables'  : [True, True],
                   'reactions'  : [True, True, True],
                   'rates'      : [True, True],
                   'events'     : [True, False, True]
                   }

new_data = rename_delete(child_name, all_data['m0'], rename, delete, required_fields)
# print(new_data)

assert new_data["states"] == {'xx0'    : {'c0': 1, 'c1': 2}, 
                              'sm1.x1' : {'c0': 0, 'c1': 0}, 
                              'xx2'    : {'c0': 0, 'c1': 0}, 
                              'sm1.x3' : {'c0': 0, 'c1': 0}
                              }
assert new_data["parameters"] == {'sm1.p0' : {'c0': 0.01, 'c1': 0.01}, 
                                  'sm1.p1' : {'c0': 0.01, 'c1': 0.01}, 
                                  'sm1.p2' : {'c0': 0.01, 'c1': 0.01}
                                  }
assert new_data["reactions"] == {'sm1.r0' : ['xx0 -> sm1.x1', 'sm1.p0*xx0'], 
                                 'sm1.r1' : ['sm1.x1->', 'sm1.p1*sm1.x1']
                                 }
assert new_data["functions"] == {'sm1.f0' : ['a', 'b', 'a*b']}
assert new_data["variables"] == {'sm1.v0': '-sm1.f0(sm1.p2, xx2)', 
                                 'sm1.v1': 'sm1.f0(sm1.p2, xx2)'
                                 }
assert new_data["rates"] == {'xx2'    : 'sm1.v0', 
                             'sm1.x3' : 'sm1.v1'
                             }
  
###############################################################################
#Test flatten
###############################################################################
required_fields = {'states'     : [True, False, False],
                   'parameters' : [True, False, False],
                   'functions'  : [True, False],
                   'variables'  : [True, True],
                   'reactions'  : [True, True, True],
                   'rates'      : [True, True],
                   'events'     : [True, False, True]
                   }

#One level deep. No recursion.
parent_ref = 'm1'
flattened  = flatten(all_data, required_fields, parent_ref)

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
assert flattened['functions'] == {'sm0.f0': ['a', 'b', 'a*b'], 
                                  'sm1.f0': ['a', 'b', 'a*b']
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
assert flattened['events'] == {'sm0.ev0': {'trigger': 'sm0.x1 < 0.01', 
                                            'assign': ['sm0.x2 = 1', 
                                                       'sm0.x3 = 1'
                                                      ]
                                            }
                                }

#Zero levels deep. No recursion.
parent_ref = 'm0'
flattened = flatten(all_data, required_fields, parent_ref)
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
assert flattened['events'] == {'ev0': {'trigger': 'x1 < 0.01', 
                                        'assign': ['x2 = 1', 'x3 = 1'
                                                  ]
                                        }
                                }
assert flattened['sim_args'] == {'line_args': {'linestyle': ':'}}

#Two levels deep. One recursive call.
parent_ref = 'm2'
flattened  = flatten(all_data, required_fields, parent_ref)

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
assert flattened['functions'] == {'sm0.f0': ['a', 'b', 'a*b'], 
                                  'sm1.sm0.f0': ['a', 'b', 'a*b'], 
                                  'sm1.sm1.f0': ['a', 'b', 'a*b']
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
assert flattened['events'] == {'ev0': {'trigger': 'sm0.x1 < 0.01', 
                                        'assign': ['sm0.x2 = 1', 'sm0.x3 = 1']
                                        }
                                }

#Circular hierarchy
parent_ref = 'm3'
try:
    flattened = flatten(all_data, required_fields, parent_ref)
except:
    assert True
else:
    assert False

parent_ref = 'm4'
try:
    flattened = flatten(all_data, required_fields, parent_ref)
except:
    assert True
else:
    assert False
    