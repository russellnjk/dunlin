model = {'states': {'x0c': {'c0': 0, 'c1': 0},
                    'x0e': {'c0': 0, 'c1': 0}
                    },
         'parameters': {'k_syn_x0c': {'c0': 0.01, 'c1': 0.05},
                        'J_x0_x'   : {'c0': 0.02, 'c1': 0.02},
                        'J_x0_y'   : {'c0': 0.02, 'c1': 0.02},
                        'F_x0e_x'  : {'c0': 0.01, 'c1': 0.01},
                        'F_x0e_y'  : {'c0': 0,    'c1': 0   },
                        'deg_x0e'  : {'c0': 0.03, 'c1': 0.03},
                        'k_tr_x0'  : {'c0': 0.01, 'c1': 0.01}
                        },
         'reactions': {'tr_x0'  : ['x0c -> x0e', 'k_tr_x0*x0c', 'k_tr_x0*x0e'],
                       'syn_x0c': ['-> x0c', 'k_syn_x0c']
                       },
         'diffusion': {'x0c': ['J_x0_x', 'J_x0_y'],
                       'x0e': ['J_x0_x', 'J_x0_y']
                       },
         'advection': {'x0c': [0, 0],
                       'x0e': ['F_x0e_x', 'F_x0e_y'],
                       },
         'compartments': {'c': {'domain_type': 'cytosolic',
                                'contains'   : ['x0c'],
                                'unit_size'  : 1
                                },
                          'e': {'domain_type': 'extracellular',
                                'contains'   : ['x0e'],
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

    
