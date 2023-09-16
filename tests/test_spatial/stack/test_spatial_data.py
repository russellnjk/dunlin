
states = {'A' : [1],
          'B' : [2],
          'C' : [3],
          'D' : [4]
          }

parameters = {'k_degA' : [0.01 ],
              'k_synB' : [0.01 ],
              'k_synD' : [0.01 ],
              'k_degD' : [0.02],
              'k_pump' : [0.05],
                
              'J_B'   : [0.001],
              'J_C_x' : [0.001],
              'J_C_y' : [0.001],
              
              'F_B'   : [0.001],
              'F_C_x' : [0.001],
              'F_C_y' : [0.001],
              }

functions = {'func0': ['x', 'k', '-x*k']
             }

variables = {'vrb0' : 'func0(k_degA, A)',
             'vrb1' : 'k_pump*B*C',
             'vrb2' : 1,
             'vrb3' : 'vrb0*vrb1*vrb2'
             }

reactions = {'synB'  : {'stoichiometry' : {'B': 1}, 
                        'rate'          : 'k_synB'
                        },
             'pumpB' : {'stoichiometry' : {'B' : -1, 'C' : 1},
                        'rate'          : 'k_pump*B'
                        },
             'synD'  : {'stoichiometry' : {'C' : -1, 'D' : 1},
                        'rate'          : 'k_synD*C - vrb2*D'
                        }
             }

rates = {'A': 'vrb0'
         }

events = {'ev0': {'trigger' : 'D - 5', 
                  'assign'  : {'D' :'0'}
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

boundary_conditions = {'C' : {'x': {'min': [ 1, 'Neumann'],
                                    'max': [-1, 'Neumann']
                                    },
                              'y': {'min': [0, 'Dirichlet'],
                                    'max': [0, 'Dirichlet'],
                                    }
                              }
                       }

coordinate_components = {'x': [0, 4], 
                         'y': [0, 4]
                         }

grid_config = {'step' : 1,
               'min'  : [0, 0],
               'max'  : [4, 4],
               }

domain_types = {'extracellular' : {'states' : ['C', 'D'],
                                   'domains' :{'medium' : [0.5, 0.3]}
                                   },
                'cytosolic'     : {'states' : ['A', 'B'],
                                   'domains': {'cytosol': [1.5, 1.2]}
                                   }
                }

surfaces = {'membrane': ['cytosol', 'medium']
            }

geometry_definitions = {'cell': {'geometry_type' : 'csg',
                                 'domain_type'   : 'cytosolic',
                                 'order'         : 1,
                                 'definition'    : ['translate', 2, 2, 
                                                    'square'
                                                    ]
                                 },
                        'field': {'geometry_type' : 'csg',
                                  'domain_type'   : 'extracellular',
                                  'order'         : 0,
                                  'definition'    : ['translate', 2, 2,
                                                     ['scale', 2, 2, 
                                                     'square']
                                                     ]
                                  }
                        }

M0 = {'states'                : states,
      'parameters'            : parameters,
      'reactions'             : reactions,
      'functions'             : functions,
      'variables'             : variables,
      'rates'                 : rates,
      'advection'             : advection,
      'diffusion'             : diffusion,
      'boundary_conditions'   : boundary_conditions,
      'events'                : events,
      'coordinate_components' : coordinate_components,
      'grid_config'           : grid_config,
      'domain_types'          : domain_types,
      'surfaces'              : surfaces,
      'geometry_definitions'  : geometry_definitions
      }

M1 = {'states'                : states,
      'parameters'            : parameters,
      'reactions'             : reactions,
      'functions'             : functions,
      'variables'             : variables,
      'rates'                 : rates,
      'advection'             : advection,
      'diffusion'             : diffusion,
      'boundary_conditions'   : boundary_conditions,
      'events'                : events,
      'coordinate_components' : coordinate_components,
      'grid_config'           : {'step' : 0.1,
                                 'min'  : [0, 0],
                                 'max'  : [4, 4],
                                 },
      'domain_types'          : domain_types,
      'surfaces'              : surfaces,
      'geometry_definitions'  : geometry_definitions
      }

M2 = {'states'                : states,
      'parameters'            : parameters,
      'reactions'             : reactions,
      'functions'             : functions,
      'variables'             : variables,
      'rates'                 : rates,
      'advection'             : advection,
      'diffusion'             : diffusion,
      'boundary_conditions'   : boundary_conditions,
      'events'                : events,
      'coordinate_components' : coordinate_components,
      'grid_config'           : {'step' : 0.01,
                                 'min'  : [0, 0],
                                 'max'  : [4, 4],
                                 },
      'domain_types'          : domain_types,
      'surfaces'              : surfaces,
      'geometry_definitions'  : geometry_definitions
      }

all_data = {'M0': M0, 'M1': M1, 'M2': M2}