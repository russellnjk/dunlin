import addpath
import dunlin.utils as ut
from dunlin.comp.child import make_child_item
from dunlin.comp.flatten import delete_rename, flatten
from data import all_data

def rename_x(x_name, x_data, child_name, rename, delete):
    new_name = make_child_item(x_name, child_name, rename, delete)
    return new_name, x_data

def rename_p(p_name, p_data, child_name, rename, delete):
    new_name = make_child_item(p_name, child_name, rename, delete)
    return new_name, p_data

def rename_func(func_name, func_data, child_name, rename, delete):
    new_name = make_child_item(func_name, child_name, rename, delete)
    f        = lambda v: make_child_item(v, child_name, rename, delete)
    new_data = [f(v) for v in func_data]
    return new_name, new_data

def rename_vrb(vrb_name, vrb_data, child_name, rename, delete):
    new_name = make_child_item(vrb_name, child_name, rename, delete)
    new_data = make_child_item(vrb_data, child_name, rename, delete)
    return new_name, new_data

def rename_rt(rt_name, rt_data, child_name, rename, delete):
    new_name = make_child_item(rt_name, child_name, rename, delete)
    new_data = make_child_item(rt_data, child_name, rename, delete)
    return new_name, new_data

def rename_rxn(rxn_name, rxn_data, child_name, rename, delete):
    new_name = make_child_item(rxn_name, child_name, rename, delete)
    
    f = lambda v: make_child_item(v, child_name, rename, delete)
    if ut.isdictlike(rxn_data):
        new_data = {k: f(v) for k, v in rxn_data.items()}
    elif ut.islistlike(rxn_data):
        new_data = [f(v) for v in rxn_data]
    
    return new_name, new_data

def rename_ev(ev_name, ev_data, child_name, rename, delete):
    new_name = make_child_item(ev_name, child_name, rename, delete)
    
    f = lambda v: make_child_item(v, child_name, rename, delete)
    g = lambda v: [f(i) for i in v] if ut.islistlike(v) else f(v)
    if ut.isdictlike(ev_data):
        
        new_data = {k: g(v) for k, v in ev_data.items()}
    elif ut.islistlike(ev_data):
        new_data = [g(v) for v in ev_data]
    
    return new_name, new_data
    

required_fields = {'states'     : rename_x,
                   'parameters' : rename_p,
                   'functions'  : rename_func,
                   'variables'  : rename_vrb,
                   'rates'      : rename_rt,
                   'reactions'  : rename_rxn,
                   'events'     : rename_ev
                   }


parent_ref  = 'm1'
parent_data = all_data[parent_ref]
child_name  = 'sm0'

child_config = parent_data['submodels'][child_name]
child_ref    = child_config['ref']
delete       = child_config.get('delete', [])
rename       = child_config.get('rename', [])

child_data      = all_data[child_ref]

submodel_data = delete_rename(child_name, child_data, rename, delete, required_fields) 

assert submodel_data['states'] == {'sm0.x0': {'c0': 1, 'c1': 2}, 
                                    'sm0.x1': {'c0': 0, 'c1': 0}, 
                                    'sm0.x2': {'c0': 0, 'c1': 0}, 
                                    'sm0.x3': {'c0': 0, 'c1': 0}
                                    }
assert submodel_data['parameters'] == {'sm0.p0': {'c0': 0.01, 'c1': 0.01}, 
                                        'sm0.p1': {'c0': 0.01, 'c1': 0.01}, 
                                        'sm0.p2': {'c0': 0.01, 'c1': 0.01}
                                        }
assert submodel_data['functions'] == {'sm0.f0': ['sm0.a', 'sm0.b', 'sm0.a*sm0.b']}
assert submodel_data['variables'] == {'sm0.v0': '-sm0.f0(sm0.p2, sm0.x2)', 
                                      'sm0.v1': 'sm0.f0(sm0.p2, sm0.x2)'
                                      }
assert submodel_data['rates'] == {'sm0.x2': 'sm0.v0', 
                                  'sm0.x3': 'sm0.v1'
                                  }
assert submodel_data['reactions'] == {'sm0.r0': ['sm0.x0 -> sm0.x1', 'sm0.p0*sm0.x0'], 
                                      'sm0.r1': ['sm0.x1->', 'sm0.p1*sm0.x1']
                                      }
assert submodel_data['events'] == {'sm0.ev0': {'trigger': 'time==0', 
                                                'assign': ['sm0.x2 = 1', 
                                                          'sm0.x3 = 1'
                                                          ]
                                                }
                                    }

###############################################################################
#Test flatten
###############################################################################
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
assert flattened['events'] == {'ev0': {'trigger': 'time==0', 
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
    