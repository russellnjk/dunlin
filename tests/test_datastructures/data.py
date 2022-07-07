m0 = {'states'    : {'x0': {'c0': 1, 'c1': 2}, 
                     'x1': {'c0': 0, 'c1': 0}, 
                     'x2': {'c0': 0, 'c1': 0},
                     'x3': {'c0': 0, 'c1': 0},
                     },
      'parameters': {'p0': {'c0': .01, 'c1': 0.01}, 
                     'p1': {'c0': .01, 'c1': 0.01},
                     'p2': {'c0': .01, 'c1': 0.01}
                     },
      'reactions' : {'r0': ['x0 -> x1', 'p0*x0'],
                     'r1': ['x1->',     'p1*x1']
                     },
      'functions' : {'f0': ['a', 'b','a*b']},
      'variables' : {'v0': '-f0(p2, x2)',
                     'v1': 'f0(p2, x2)'
                     },
      'rates'     : {'x2': 'v0',
                     'x3': 'v1'
                     },
      'events'    : {'ev0': {'trigger': 'time==0', 
                             'assign': ['x2 = 1', 
                                        'x3 = 1'
                                        ]
                             }
                     },
      'sim_args'  : {'line_args': {'linestyle': ':'
                                   }
                     }
      }

m1 = {'states'   :{'sm0.x0':{'c0': 3, 'c1': 2}
                   },
      'submodels': {'sm0': {'ref': 'm0'
                            },
                    'sm1': {'ref': 'm0',
                            'delete': ['ev0'],
                            'rename': {'x0': 'xx0', 'x2': 'xx2'}
                            }
                    }
      }


m2 = {'states'    : {'sm0.sm0.x0' : {'c0': 3, 'c1': 2},
                     },
      'submodels' : {'sm0': {'ref'   : 'm0',
                             'rename': {'ev0': 'ev0'}
                             },
                     'sm1': {'ref'   : 'm1',
                             'delete': ['sm0.ev0']
                             }
                     }
      }

m3 = {'states'     :{'x0': [0]
                     },
      'parameters' : {'p0': [0.01]
                      },
      'submodels'  : {'sm0': {'ref': 'm0'
                              },
                      'sm1': {'ref': 'm4'
                              },
                      }
      }

m4 = {'submodels' : {'sm0': {'ref'   : 'm3',
                             },
                     }
      }

all_data = {'m0': m0, 'm1': m1, 'm2': m2, 'm3': m3, 'm4': m4}

