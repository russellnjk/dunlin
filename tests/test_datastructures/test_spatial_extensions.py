import addpath
import dunlin                            as dn
import dunlin.utils                      as ut
import dunlin.standardfile.dunl.readdunl as rdn
from dunlin.datastructures.reaction            import Reaction,            ReactionDict
from dunlin.datastructures.gridconfig          import GridConfig,          GridConfigDict
from dunlin.datastructures.domaintype          import DomainType,          DomainTypeDict
from dunlin.datastructures.geometrydefinition  import GeometryDefinition,  GeometryDefinitionDict
from dunlin.datastructures.boundarycondition   import BoundaryConditions,  BoundaryConditionDict
from dunlin.datastructures.compartment         import Compartment,         CompartmentDict
from dunlin.datastructures.masstransfer        import (Advection, AdvectionDict, 
                                                       Diffusion, DiffusionDict
                                                       )

from dunlin.datastructures.coordinatecomponent import CoordinateComponentDict
from dunlin.datastructures.stateparam          import ParameterDict, StateDict

data0 = {'x': [0, 10],
         'y': [0, 10]
         }

ccd = CoordinateComponentDict(data0)


ext_namespace = set()
data0 = {'dmnt0': {'dmn0': [2, 2]
                   },
         'dmnt1': {'dmn1': [5, 5]
                   }
          }

dmnts         = DomainTypeDict(ext_namespace, ccd, data0)

data0  = {'x0c': [1], 'x0e': [1]}
xs     = StateDict(ext_namespace, data0)

data0 = {'k0': [0.05], 'k1': [0.005], 
         'J_x0_x': [0.02], 'J_x0_y': [0.03],
         'F_x0_x': [0.02], 'F_x0_y': [0.03],
         }
ps    = ParameterDict(ext_namespace, data0)

data0 = {'trans': ['x0c -> x0e', 'k0*x0c', 'k0*x0e'],
         'syn'  : ['-> x0c', 'k1']
         }
rxns  = ReactionDict(ext_namespace, data0)

###############################################################################
#Test Compartment
###############################################################################
data0 = {'cpt0': {'domain_type': 'dmnt1',
                  'contains'   : ['x0c']
                  },
         'cpt1': {'domain_type': 'dmnt0',
                  'contains'   : ['x0e']
                  },
         }

C = CompartmentDict

#Test instantiation
F0 = C(ext_namespace, xs, dmnts, data0)

data0_ = {'cpt0': {'domain_type': 'dmnt1',
                  'contains'   : ['x0c']
                  },
         'cpt1': {'domain_type': 'dmnt0',
                  'contains'   : ['x2e']
                  },
         }

try:
    F0 = C(set(), xs, dmnts, data0_)
except:
    assert True
else:
    assert False
    
data0_ = {'cpt0': {'domain_type': 'dmnt2',
                  'contains'   : ['x0c']
                  },
         'cpt1': {'domain_type': 'dmnt0',
                  'contains'   : ['x0e']
                  },
         }

try:
    F0 = C(set(), xs, dmnts, data0_)
except:
    assert True
else:
    assert False
    
#Test access
f0 = F0['cpt0']

#Test export/roundtrip
data1 = F0.to_data()
dunl = F0.to_dunl()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0 

###############################################################################
#Test Advection
###############################################################################
data0 = {'x0e': ['F_x0_x', 'F_x0_y'],
         }

C = AdvectionDict

#Test instantiation
F0 = C(ext_namespace, ccd, None, xs, ps, data0)

data0_ = {'x0e': ['F_x0_x', 'F_x0_y', 'F_x0_y'],
          }

try:
    F0 = C(set(), ccd, None, xs, ps, data0_)
except:
    assert True
else:
    assert False
    
#Test access
f0 = F0['x0e']

#Test export/roundtrip
data1 = F0.to_data()
dunl = F0.to_dunl()
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
F0 = C(ext_namespace, ccd, None, xs, ps, data0)

data0_ = {'x0c': ['J_x0_x', 'J_x0_y'],
          'x0d': 'J_x0_x',
          }

try:
    F0 = C(set(), ccd, None, xs, ps, data0_)
except:
    assert True
else:
    assert False
    
#Test access
f0 = F0['x0e']

#Test export/roundtrip
data1 = F0.to_data()
dunl = F0.to_dunl()
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
F0 = C(ext_namespace, ccd, xs, ps, data0)

data0 = {'x0c': {'x': {'min': [0, 'Neumann'],
                       'max': [0, 'Neumann']
                       }
                 }
         }

F0 = C(ext_namespace, ccd, xs, ps, data0)

data0_ = {'x0c': {'x': {'min': [0, 'Neumann'],
                        'max': [0, 'dna']
                       }
                 }
         }

try:
    F0 = C(ext_namespace, ccd, xs, ps, data0_)
except:
    assert True
else:
    assert False
    
#Test access
f0 = F0['x0c']

#Test export/roundtrip
data1 = F0.to_data()
dunl = F0.to_dunl()
data2 = rdn.read_dunl_code(';A\n' + dunl)['A']
assert data2 == data1 == data0
    