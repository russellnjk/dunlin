import addpath
import dunlin                            as dn
import dunlin.utils                      as ut
import dunlin.standardfile.dunl.readdunl as rdn
from dunlin.datastructures.coordinatecomponent import CoordinateComponentDict
from dunlin.datastructures.gridconfig          import GridConfig,          GridConfigDict
from dunlin.datastructures.domain              import Domain,              DomainDict
from dunlin.datastructures.geometrydefinition  import GeometryDefinition,  GeometryDefinitionDict
from dunlin.datastructures.adjacentdomain      import AdjacentDomainDict
from dunlin.datastructures.compartment         import CompartmentDict

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
#Test Domain
###############################################################################
data0 = {'dmn0': {'compartment'    : 'cpt0',
                  'internal_point' : [2, 2, 2]
                   },
          'dmn0': {'compartment'    : 'cpt1',
                   'internal_point' : [2, 2, 3]
                   },
          }

C = DomainDict

compartments = CompartmentDict(set(), xs, {'cpt0': ['x0']})

#Test instantiation
F0 = C(all_names, ccd, data0)

data0_ = {'dmnt0': {'dmn0': [[2, 2, ]]
                    },
          'dmnt1': {'dmn0': [[2, 2, 2]]
                    }
          }
try:
    F0 = C(set(), ccd, data0_)
except:
    assert True
else:
    assert False

data0_ = {'dmnt0': {'dmn0': [[2, 2, 2]]
                    },
          'dmnt1': {'dmn0': [[5, 5, 5]]
                    }
          }
try:
    F0 = C(set(), ccd, data0_)
except:
    assert True
else:
    assert False

#Test access
f0 = F0['dmnt0']

#Test export/roundtrip
data1 = F0.to_dict()
dunl = F0.to_dunl_elements()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0 

dmnts = F0

###############################################################################
#Test AdjacentDomains
###############################################################################
data0 = {'interface': ['dmn0', 'dmn1']}

C = AdjacentDomainDict

#Test instantiation
F0 = C(all_names, ccd, dmnts, data0)

data0_ = {'interface': ['dmn0', 'dmn2']}

try:
    F0 = C(all_names, ccd, dmnts, data0_)
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
data0 = {'gr_main': {'config' : [0.02, [0, 10], [0, 10], [0, 10]], 'children': ['gr0']},
         'gr0'    : {'config' : [0.01, [4, 6],  [4, 6],  [4, 6]]}
         }

C = GridConfigDict

#Test instantiation
F0 = C(all_names, ccd, data0)

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
sphere     = ['sphere', ['translate', 5, 5, 5]]
cube       = ['cube',   ['translate', 5, 5, 6]]
hemisphere = ['difference', sphere, cube]
tank       = ['cube', ['scale', 10, 10, 10], ['translate', 5, 5, 5]]

data0 = {'hemisphere': {'definition': 'csg',
                        'domain_type': 'dmnt0',
                        'order': 1,
                        'node': hemisphere
                        },
          'tank'     : {'definition': 'csg',
                        'domain_type': 'dmnt0',
                        'order': 0,
                        'node': tank
                        }
          }

C = GeometryDefinitionDict

#Test instantiation
F0 = C(all_names, ccd, dmnts, data0)

data0_ = {'hemisphere': {'definition': 'csg',
                          'domain_type': 'dmnt0',
                          'order': 0,
                          'node': hemisphere
                          },
          'tank'     : {'definition': 'csg',
                        'domain_type': 'dmnt0',
                        'order': 0,
                        'node': tank
                        }
          }

try:
    F0 = C(set(), ccd, dmnts, data0_)
except:
    assert True
else:
    assert False

data0_ = {'hemisphere': {'definition': 'csg',
                          'domain_type': 'dmnt0',
                          'order': 1,
                          'node': hemisphere
                          },
          'tank'     : {'definition': 'csg',
                        'domain_type': 'dmnt0',
                        'order': 0,
                        'node': tank,
                        'extra': 'haha'
                        }
          }

try:
    F0 = C(set(), ccd, dmnts, data0_)
except:
    assert True
else:
    assert False
    
#Test access
f0 = F0['hemisphere']

#Test export/roundtrip
data1 = F0.to_dict()
dunl = F0.to_dunl_elements()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0 

