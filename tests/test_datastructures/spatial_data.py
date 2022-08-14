model = {'states': {'x0c': [0, 0],
                    'x0e': [0, 0]
                    },
         'parameters': {'k_syn_x0c': [0.01, 0.05],
                        'J_x0_x'   : [0.02, 0.02],
                        'J_x0_y '  : [0.02, 0.02],
                        'F_x0e_x'  : [0.01, 0.01],
                        'F_x0e_y'  : [0,    0   ],
                        'deg_x0e'  : [0.03, 0.03],
                        'tr_x0'    : [0.01, 0.01]
                        },
         'reactions': {'tr_x0'  : ['x0c -> x0e', 'tr_x0*x0c', 'tr_x0*x0e'],
                       'syn_x0c': ['    -> x0c', 'k_syn_x0c']
                       },
         'diffusion': {'dfn_x0c': ['x0c', 'J_x0_x', 'J_x0_y'],
                       'dfn_x0e': ['x0e', 'J_x0_x', 'J_x0_y']
                       },
         'advection': {'adv_x0e': ['x0e', 'F_x0e_x', 'F_x0e_y'],
                       },
         'compartments': {'c': {'domain_type': 'cytosolic',
                                'contains'   : ['x0c'],
                                'unit_size'  : 1
                                },
                          'e': {'domain_type': 'extracellular',
                                'contains'   : 'x0e',
                                'unit_size'  : 1
                                }
                          }
         }

geometry_definitions = {'well': {'definition'  : 'csg',
                                 'domain_type' : 'extracellular',
                                 'order'       : 0,
                                 'node'        : ['square', ['scale', 10, 8]]
                                 },
                        'cell': {'definition' : 'csg',
                                 'domain_type': 'cytosolic',
                                 'order'      : 1,
                                 'node'       : ['circle', ['scale', 1.2, 1.2]]
                                 },
                        }

geometry = {'coordinate_components': {'x': [-10, 10],
                                      'y': [-10, 10]
                                      },
            'grid_config': {'gr_main': {'config'  : [0.1,  [-10, 10], [-10, 10]],
                                        'children': ['gr0']
                                        },
                            'gr0'    : {'config'  : [0.05,  [-5, 5 ], [-5,  5 ]],
                                        'children': ['gr1']
                                        },
                            'gr1'    : {'config'  : [0.025, [-2, 2],  [-2,  2 ]],
                                        }
                            },
            'domain_types': {'cytosolic': {'domains': {'cytosol0': [[0, 0]]
                                                       },
                                           'ndims'  : 2,
                                           },
                             'extracellular': {'domains': {'medium0': [[-8, 0]]
                                                                        },
                                                            'ndims'  : 2,
                                                            },
                             },
            'adjacent_domains': {'cytosol_medium': ['cytosol0', 'medium0'],
                                 },
            'geometry_definitions': geometry_definitions
            }

all_data = {'M0': model, 'Geo0': geometry}

    
