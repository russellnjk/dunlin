m1 = {'states'     : {'x0' : [0, 100],
                      'x1' : [0, 100]
                      },
      'parameters' : {'p0' : [0.1]*2,
                      'p1' : [0.1]*2,
                      'u0' : [1  ]*2,
                      'u1' : [9  ]*2
                      },
      'reactions' : {'g0' : {'stoichiometry' : {'x0': 1},
                             'rate'          : 'u0'
                             },
                     'g1' : {'stoichiometry' : {'x1': 1},
                             'rate'          : 'u1'
                             },
                     'r0' : {'stoichiometry' : {'x0': -1},
                             'rate'          : 'p0*x0'
                             },
                     'r1' : {'stoichiometry' : {'x1': -1},
                             'rate'          : 'p1*x1'
                             },
                     },
      'variables' : {'v0' : 'x0'
                     },
      'tspans'    : {0: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                     1: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                     },
      'optim_args': {'free_parameters' : {'u0': {'bounds': [0, 10], 
                                                 },
                                          'u1': {'bounds': [0, 10], 
                                                 },
                                          },
                     'settings'   : {'disp'   : False,
                                     'popsize': 5
                                     },
                     'line_args'  : {'color': {0: 'steel'}, 
                                     'marker': '+'
                                     }
                     },
      'sim_args': {'line_args': {'color': {0: 'coral',
                                           1: 'dark yellow'
                                           }
                                 }
                   } 
      }

all_data = {'M1': m1}
