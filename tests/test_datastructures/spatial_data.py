
states = {'A' : [0, 1],
          'B' : [1, 2],
          'C' : [0, 3],
          'D' : [0, 4]
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

reactions = {'synB'  : ['-> B', 'k_synB'],
             'pumpB' : ['B -> C', 'k_pump*B'],
             'synD'  : ['C -> D', 'k_synD*C', 'vrb2*D']
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

compartments = {'cytosol' : {'domain_type': 'cytosolic',
                             'contains'   : ['A', 'B'],
                             'unit_size'  : 1
                             },
                'medium' : {'domain_type': 'extracellular',
                            'contains'   : ['C', 'D'],
                            'unit_size'  : 1
                            }
                }

# boundary_conditions = {'bc_C_x_min' : ['C',  1, 'Neumann',   'x', 'min'],
#                        'bc_C_x_max' : ['C', -1, 'Neumann',   'x', 'max'],
#                        'bc_C_y'     : ['C', 0, 'Dirichlet', 'y']
#                        }

boundary_conditions = {'C': {'xmin' : [ 1, 'Neumann'  ],
                             'xmax' : [-1, 'Neumann'  ],
                             'ymin' : [ 0, 'Dirichlet']
                             },
                       }

coordinate_components = {'x': [0, 4], 
                         'y': [0, 4]
                         }

grid_config = {'step' : 1,
               'min'  : [0, 0],
               'max'  : [4, 4],
               }

domain_types = {'extracellular' : {'medium' : [0.5, 0.3]},
                'cytosolic'     : {'cytosol': [1.5, 1.2]}
                }

adjacent_domains = {'membrane': ['cytosol', 'medium']
                    }

geometry_definitions = {'cell': {'definition' : 'csg',
                                 'domain_type': 'cytosolic',
                                 'order'      : 1,
                                 'node'       : ['square', 
                                                 ['translate', 2, 2]
                                                 ]
                                 },
                        'field': {'definition' : 'csg',
                                  'domain_type': 'extracellular',
                                  'order'      : 0,
                                  'node'       : ['square', 
                                                  ['scale', 2, 2], 
                                                  ['translate', 2, 2]
                                                  ]
                                  }
                        }

geometry = {'coordinate_components': coordinate_components,
            'grid_config'          : grid_config,
            'domain_types'         : domain_types,
            'adjacent_domains'     : adjacent_domains,
            'geometry_definitions' : geometry_definitions
            }

M0 = {'states'              : states,
      'parameters'          : parameters,
      'reactions'           : reactions,
      'functions'           : functions,
      'variables'           : variables,
      'rates'               : rates,
      'diffusion'           : diffusion,
      'advection'           : advection,
      'compartments'        : compartments,
      'boundary_conditions' : boundary_conditions,
      'geometry'            : geometry,
      'events'              : events 
      }

all_data = {'M0': M0}