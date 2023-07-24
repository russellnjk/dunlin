import addpath
import dunlin                   as dn
import dunlin.standardfile.dunl as sfd
from dunlin.datastructures.ode          import ODEModelData
from dunlin.datastructures.spatial      import SpatialModelData 

states = {'A' : [0, 1],
          'B' : [1, 2],
          'C' : [0, 3],
          'D' : [0, 4],
          'E' : [5, 5]
          }

parameters = {'k_degA' : [0.1, 0.1  ],
              'k_synB' : [1,   1    ],
              'k_synD' : [0.1, 0.1  ],
              'k_degD' : [0.02, 0.02],
              'k_pump' : [0.05, 0.05],
                
              'J_B'   : [1, 0.1],
              'J_C_x' : [0, 0.1],
              'J_C_y' : [1, 0.1],
              
              'F_B'   : [1, 0.1],
              'F_C_x' : [0, 0.1],
              'F_C_y' : [1, 0.1],
              }

functions = {'func0': ['x', 'k', '-x*k']
             }

variables = {'vrb0' : 'func0(k_degA, A)',
             'vrb1' : 'k_pump*B*C',
             'vrb2' : 1,
             'vrb3' : 'vrb0*vrb1*vrb2'
             }

reactions = {'synB'  : [{'B': 1}, 'k_synB'],
             'pumpB' : [{'B': -1, 'C': -1}, 'k_pump*B'],
             'synD'  : [{'C': -1, 'D': 1}, 'k_synD*C - vrb2*D']
             }

rates = {'A': 'vrb0'
         }

events = {'ev0': {'trigger' : 'D > 5', 
                  'assign'  : ['D = 0']
                  }
          }

diffusion = {'B' : 'J_B',
             'C' : ['J_C_x', 'J_C_y'],
             'D' : 1
             }

advection = {'B' : 'F_B',
             'C' : ['F_C_x', 'F_C_y'],
             'D' : 1
             }

boundary_conditions = {'C': {'x' : {'min' : [1, 'Neumann'  ],       
                                    'max' : [-1, 'Neumann' ]
                                    },
                             'y' : {'min' : [ 0, 'Dirichlet']
                                    }
                             },
                       }

coordinate_components = {'x': [0, 4], 
                         'y': [0, 4]
                         }

grid_config = {'step' : 1,
               'min'  : [0, 0],
               'max'  : [4, 4],
               }

domain_types = {'cytosol' : {'states'  : ['A', 'B'],
                             'domains' : {'cytosol0' : [3, 3]}
                             },
                'medium'  : {'states'  : ['C', 'D'],
                             'domains' : {'medium0' : [2, 3]}
                             },
                }

#No states associated with any surfaces
surfaces = {'membrane0': ['cytosol0', 'medium0']
            }

geometry_definitions = {'cell': {'geometry_type' : 'csg',
                                 'domain_type'   : 'cytosol',
                                 'order'         : 1,
                                 'definition'    : ['translate', 2, 2, 'square']
                                 },
                        'field': {'geometry_type' : 'csg',
                                  'domain_type'   : 'medium',
                                  'order'         : 0,
                                  'definition'    : ['translate', 2, 2, ['scale', 2, 2, 'square']]
                                  }
                        }

M0 = {'states'                : states,
      'parameters'            : parameters,
      'reactions'             : reactions,
      'functions'             : functions,
      'variables'             : variables,
      'rates'                 : rates,
      'diffusion'             : diffusion,
      'advection'             : advection,
      'boundary_conditions'   : boundary_conditions,
      'coordinate_components' : coordinate_components,
      'grid_config'           : grid_config,
      'domain_types'          : domain_types,
      'surfaces'              : surfaces,
      'geometry_definitions'  : geometry_definitions,
      'events'                : events 
      }

all_data = {'M0': M0}

#Test spatial data
ref = 'M0'

spldata = SpatialModelData.from_all_data(all_data, ref)
d0 = spldata.to_dict()
d1 = spldata.to_dunl_dict()

dunl = sfd.write_dunl_code(d1)
a    = sfd.read_dunl_code(dunl)

for k, v in d0.items():
    if k == 'ref':
        continue
    
    for k, v in d0.items():
        if k == 'ref':
            continue
        
        assert a[ref][k] == v
