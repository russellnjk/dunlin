import addpath
import dunlin                            as dn
import dunlin.utils                      as ut
import dunlin.standardfile.dunl.readdunl as rdn
from dunlin.datastructures.coordinatecomponent import CoordinateComponentDict
from dunlin.datastructures.gridconfig          import GridConfig,          GridConfigDict
from dunlin.datastructures.domaintype          import DomainType,          DomainTypeDict
from dunlin.datastructures.geometrydefinition  import GeometryDefinition,  GeometryDefinitionDict
from dunlin.datastructures.adjacentdomain      import AdjacentDomainsDict
 
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
data1 = F0.to_data()
dunl = F0.to_dunl()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0

ccd = F0
ext_namespace = set()

###############################################################################
#Test GridConfig
###############################################################################
data0 = {'gr_main': {'config' : [0.02, [0, 10], [0, 10], [0, 10]], 'children': ['gr0']},
         'gr0'    : {'config' : [0.01, [4, 6],  [4, 6],  [4, 6]]}
         }

C = GridConfigDict

#Test instantiation
F0 = C(ext_namespace, ccd, data0)

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
data1 = F0.to_data()
dunl = F0.to_dunl()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0 

###############################################################################
#Test DomainType
###############################################################################
data0 = {'dmnt0': {'dmn0': [2, 2, 2]
                   },
          'dmnt1': {'dmn1': [5, 5, 5]
                    }
          }

C = DomainTypeDict

#Test instantiation
F0 = C(ext_namespace, ccd, data0)

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
data1 = F0.to_data()
dunl = F0.to_dunl()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0 

dmnts = F0

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
F0 = C(ext_namespace, ccd, dmnts, data0)

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
data1 = F0.to_data()
dunl = F0.to_dunl()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0 

###############################################################################
#Test AdjacentDomains
###############################################################################
data0 = {'interface': ['dmn0', 'dmn1']}

C = AdjacentDomainsDict

#Test instantiation
F0 = C(ext_namespace, ccd, dmnts, data0)

data0_ = {'interface': ['dmn0', 'dmn2']}

try:
    F0 = C(ext_namespace, ccd, dmnts, data0_)
except:
    assert True
else:
    assert False
    
#Test access
f0 = F0['interface']

#Test export/roundtrip
data1 = F0.to_data()
dunl = F0.to_dunl()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0