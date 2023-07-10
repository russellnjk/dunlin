import addpath
import dunlin                            as dn
import dunlin.utils                      as ut
import dunlin.standardfile.dunl.readdunl as rdn
from dunlin.datastructures.reaction            import Reaction,            ReactionDict
from dunlin.datastructures.gridconfig          import GridConfig          #GridConfigDict
from dunlin.datastructures.geometrydefinition  import GeometryDefinition,  GeometryDefinitionDict
from dunlin.datastructures.boundarycondition   import BoundaryConditions,  BoundaryConditionDict

from dunlin.datastructures.compartment         import Compartment,         CompartmentDict
from dunlin.datastructures.domain              import Domain,              DomainDict
from dunlin.datastructures.adjacentdomain      import AdjacentDomainDict

from dunlin.datastructures.masstransfer        import (Advection, AdvectionDict, 
                                                       Diffusion, DiffusionDict
                                                       )

from dunlin.datastructures.coordinatecomponent import CoordinateComponentDict
from dunlin.datastructures.stateparam          import ParameterDict, StateDict

###############################################################################
#Test Coordinate Component
###############################################################################
data0 = {'x': [0, 10],
         'y': [0, 10],
         'z': [0, 10]
         }

C = CoordinateComponentDict

#Test instantiation
F0 = C(data0)

data0_ = {'x': [0, 10],
          'y': [0, 10],
          'w': [0, 10]
          }
try:
    F0 = C(data0_)
except:
    assert True
else:
    assert False
    
#Test access
f0 = F0['x']

#Test export/roundtrip
data1 = F0.to_dict()
dunl = F0.to_dunl_elements()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0

ccd = F0
all_names = set()

###############################################################################
#Create stuff for downstream tests
###############################################################################
data0 = {'x': [0, 10],
         'y': [0, 10]
         }

ccd = CoordinateComponentDict(data0)


all_names = set()
data0 = {'dmnt0': {'dmn0': [2, 2]
                   },
         'dmnt1': {'dmn1': [5, 5]
                   }
          }

data0  = {'x0c': [1], 'x0e': [1]}
xs     = StateDict(all_names, data0)

data0 = {'k0': [0.05], 'k1': [0.005], 
         'J_x0_x': [0.02], 'J_x0_y': [0.03],
         'F_x0_x': [0.02], 'F_x0_y': [0.03],
         }
ps    = ParameterDict(all_names, data0)

data0 = {'trans': ['x0c -> x0e', 'k0*x0c', 'k0*x0e'],
         'syn'  : ['-> x0c', 'k1']
         }
rxns  = ReactionDict(all_names, data0)

###############################################################################
#Test Compartment
###############################################################################
data0 = {'cpt0': ['x0c'],
         'cpt1': ['x0e'],
         }

C = CompartmentDict

#Test instantiation
F0 = C(all_names, xs, data0)

data0_ = {'cpt0': ['x0c'],
          'cpt1': ['x0x'],
          }

try:
    F0 = C(set(), xs, data0_)
except:
    assert True
else:
    assert False
    
#Test access
f0 = F0['cpt0']

#Test export/roundtrip
data1 = F0.to_dict()
dunl = F0.to_dunl_elements()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0 

###############################################################################
#Create Stuff for downstream tests
###############################################################################
cpts = F0

###############################################################################
#Test Domain
###############################################################################
data0 = {'dmn0': {'compartment'    : 'cpt0',
                  'internal_point' : [2, 2]
                   },
         'dmn1': {'compartment'    : 'cpt1',
                  'internal_point' : [2, 3]
                  },
          }

C = DomainDict

#Test instantiation
F0 = C(all_names, ccd, cpts, data0)

data0_ = {'dmn0': {'compartment'    : 'cpt0',
                  'internal_point' : [2, 2, 2]
                   },
          'dmn1': {'compartment'    : 'cpt1',
                   'internal_point' : [2, 3]
                   },
          }

try:
    F0 = C(set(), ccd, data0_)
except:
    assert True
else:
    assert False

#Test access
f0 = F0['dmn0']

#Test export/roundtrip
data1 = F0.to_dict()
dunl = F0.to_dunl_elements()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0 

###############################################################################
#Create Stuff for downstream tests
###############################################################################
dmns = F0

###############################################################################
#Test AdjacentDomains
###############################################################################
data0 = {'interface': ['dmn0', 'dmn1']}

C = AdjacentDomainDict

#Test instantiation
F0 = C(all_names, ccd, dmns, data0)

data0_ = {'interface': ['dmn0', 'dmn2']}

try:
    F0 = C(all_names, ccd, dmns, data0_)
except:
    assert True
else:
    assert False
    
#Test access
f0 = F0['interface']

#Test export/roundtrip
data1 = F0.to_dict()
dunl = F0.to_dunl_elements()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0

###############################################################################
#Test GridConfig
###############################################################################
data0 = {'step'     : 0.02,
         'min'      : [0,   0],
         'max'      : [10, 10],
         'children' : {'gr0': {'step': 0.01,
                               'min' : [4, 4],
                               'max' : [6, 6]
                               }
                       }
         }
# data0 = {'gr_main': {'config' : [0.02, [0, 10], [0, 10], [0, 10]], 'children': ['gr0']},
#          'gr0'    : {'config' : [0.01, [4, 6],  [4, 6],  [4, 6]]}
#          }

C = GridConfig

#Test instantiation
F0 = C(all_names, name=None, coordinate_components=ccd,**data0)

data0_ = {'gr_main': {'config' : [0.02, [0, 10], [0, 10], ], 'children': ['gr0']},
          'gr0'    : {'config' : [0.01, [4, 6], [4, 6],   [4, 6]]}
          }
try:
    F0 = C(set(), ccd, data0_)
except:
    assert True
else:
    assert False

data0_ = {'gr_main': {'config' : [0.02, [0, 10], [0, 10], [0, 10]], 'children': ['gg']},
          'gr0'    : {'config' : [0.01, [4, 6],  [4, 6],  [4, 6]]}
          }
try:
    F0 = C(set(), ccd, data0_)
except:
    assert True
else:
    assert False
    
#Test access
f0 = F0['gr0']

#Test export/roundtrip
data1 = F0.to_dict()
dunl = F0.to_dunl_elements()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0 

###############################################################################
#Test GeometryDefinition
###############################################################################
square  = ['translate', 5, 5, 'square']
circle  = ['translate', 5, 5, 'circle']
myshape = ['difference', circle, square]
tank    = ['translate', 5, 5,  
           ['scale', 10, 10, 'square']
           ]

data0 = {'myshape': {'geometry_type': 'csg',
                     'compartment'  : 'cpt0',
                     'order'        : 1,
                     'definition'   : myshape
                     },
          'tank'  : {'geometry_type': 'csg',
                     'compartment'  : 'cpt0',
                     'order'        : 0,
                     'definition'   : tank
                     }
          }

C = GeometryDefinitionDict

#Test instantiation
F0 = C(all_names, ccd, cpts, data0)

data0_ = {'myshape': {'geometry_type': 'csg',
                      'compartment'  : 'cpt0',
                      'order'        : 1,
                      'definition'   : myshape
                      },
          'tank'   : {'geometry_type': 'csg',
                      'compartment'  : 'cpt0',
                      'order'        : 1,
                      'definition'   : tank
                      }
          }

try:
    F0 = C(all_names, ccd, cpts, data0)
except:
    assert True
else:
    assert False

#Test access
f0 = F0['myshape']

#Test export/roundtrip
data1 = F0.to_dict()
dunl = F0.to_dunl_elements()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0 

###############################################################################
#Test Advection
###############################################################################
data0 = {'x0e': ['F_x0_x', 'F_x0_y'],
         }

C = AdvectionDict

#Test instantiation
F0 = C(all_names, ccd, xs, ps, data0)

data0_ = {'x0e': ['F_x0_x', 'F_x0_y', 'F_x0_y'],
          }

try:
    F0 = C(set(), ccd, None, xs, ps, data0_)
except:
    assert True
else:
    assert False
    
#Test access
f0 = F0['x0e', 'x']

#Test export/roundtrip
data1 = F0.to_dict()
dunl  = F0.to_dunl_elements()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0 

###############################################################################
#Test Diffusion
###############################################################################
data0 = {'x0c': ['J_x0_x', 'J_x0_y'],
         'x0e': 'J_x0_x',
         }

C = DiffusionDict

#Test instantiation
F0 = C(all_names, ccd, xs, ps, data0)

data0_ = {'x0c': ['J_x0_x', 'J_x0_y'],
          'x0d': 'J_x0_x',
          }

try:
    F0 = C(set(), ccd, xs, ps, data0_)
except:
    assert True
else:
    assert False
    
#Test access
f0 = F0['x0e', 'x']

#Test export/roundtrip
data1 = F0.to_dict()
dunl = F0.to_dunl_elements()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0 

###############################################################################
#Test BoundaryCondition
###############################################################################
data0 = {'x0c': {'x': {'min': {'value'         : 0, 
                               'condition_type': 'Neumann'
                               },
                       'max': {'value'         : 0, 
                               'condition_type': 'Neumann'
                               }
                       }
                 }
         }

C = BoundaryConditionDict

#Test instantiation
F0 = C(all_names, ccd, xs, ps, data0)

data0 = {'x0c': {'x': {'min': [0, 'Neumann'],
                       'max': [0, 'Neumann']
                       }
                 }
         }

F0 = C(all_names, ccd, xs, ps, data0)

data0_ = {'x0c': {'x': {'min': [0, 'Neumann'],
                        'max': [0, 'dna']
                       }
                 }
         }

try:
    F0 = C(all_names, ccd, xs, ps, data0_)
except:
    assert True
else:
    assert False
    
#Test access
f0 = F0['x0c', 'x', 'min']

#Test export/roundtrip
data1 = F0.to_dict()
dunl = F0.to_dunl_elements()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0

