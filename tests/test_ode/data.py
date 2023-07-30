m0 = {'states'    : {'x0': [1], 
                     'x1': [0], 
                     'x2': [0],
                     'x3': [0]
                     },
      'parameters': {'p0': [0.01], 
                     'p1': [0.01]
                     },
      'reactions' : {'r0': {'stoichiometry' : {'x0': -1, 
                                               'x1':  2
                                               }, 
                            'rate'          : 'p0*x0'
                            }
                     },
      'functions' : {'f0': ['a', 'b', 'a*b']},
      'variables' : {'v0': '-f0(p1, x2)'},
      'rates'     : {'x2': 'v0',
                     'x3': 'v0'
                     },
      'events'    : {'ev0': {'trigger': 'time', 
                             'assign': {'x2': 1, 
                                        'x3': 1
                                        }
                             }
                     },
      }

#m1 uses m0 entirely
#Use this to test if child items are created correctly
m1 = {'states'    : {'x0' : 0
                     },
      'rates'     : {'x0': '-0.1*x0'
                     },
      'submodels' : {'m0' : {'ref'    : 'M0', 
                             }
                     }
      }

#m2 uses m0 but deletes everything
#Use this to test delete
m2 = {'states'    : {'x0' : 0
                     },
      'rates'     : {'x0': '-0.1*x0'
                     },
      'submodels' : {'m0' : {'ref'    : 'M0',
                             'delete' : ['ev0', 'x0', 'x1', 'x2', 'x3', 'r0', 
                                         'p0', 'p1', 'f0'
                                         ] 
                             }
                     }
      }

#m3 uses m0 but merges ("merges") its x1 with m0.x0
#There should be no m0.x0 in m3
#Use this to test merge
m3 = {'states'    : {'x0': 1, 
                     'x1': 0, 
                     'x2': 0,
                     'x3': 0
                     },
      'parameters': {'p0': 0.01, 
                     'p1': 0.01
                     },
      'reactions' : {'r0': ['x0 -> 2*x1', 'p0*x0']
                     },
      'functions' : {'f0(a, b)': 'a*b'},
      'variables' : {'v0': '-f0(p1, x2)'},
      'rates'     : {'x2': 'v0',
                     'x3': 'v0'
                     },
      'submodels' : {'m0': {'ref'   : 'M0',
                            'merge' : {'x0' : 'x0',
                                       'x1' : 'x1',
                                       'x2' : 'x2',
                                       'x3' : 'x3',
                                       'p0' : 'p0',
                                       'p1' : 'p1',
                                       'r0' : 'r0',
                                       'f0' : 'f0',
                                       'v0' : 'v0',
                                       }
                            }
                     }
      }

#m4 uses m0 but overrides m0.x0
#There should be no m0.x0 in m3
#Use this to test 
m3 = {'states'     : {'x0'    : 0,
                      'm0.x0' : 10
                      },
      'parameters' : {'m0.p0' : 0.2,
                      },
      'functions'  : {'m0.f0(a, b)': '0.5*a*b'
                      },
      'rates'      : {'x0'    : '-0.1*x0',
                      'm0.x2' : '-0.2*m0.x2'
                      },
      'submodels'  : {'m0' : {'ref': 'M0'
                              }
                      }
      }

#m4 uses one instance of m0 but merges everything and gets rid of x1
#Use this to test the flattening function
m4 = {'states'    : {'xx0'  : 1,
                     'm0.x2': 1,
                     'xx3'  : 2
                     },
      'parameters': {'pp0': 0.01, 
                     'pp1': 0.01
                     },
      'reactions' : {'m0.r0': ['xx0 -> ', '0.5*pp0*xx0']
                      },
      'functions' : {'ff0(a, b)': '0.5*a*b'},
      'variables' : {'vv0': '-ff0(pp1, m0.x2)'},
      'rates'     : {'m0.x2': 'vv0'},
      'submodels' : {'m0': {'ref'    : 'M0',
                            'delete' : ['x1', 'p1'],
                            'merge'  : {'x0'   : 'xx0', 
                                        'f0'   : 'ff0', 
                                        'p0'   : 'pp0',
                                        'x3'   : 'xx3'
                                        }
                            }
                      }
      }

#m5 has a circular hierarchy with m6 and m7
#Use this to test error-catching
m5 = {'states'    : {'x0' : 0
                     },
      'rates'     : {'x0': '-0.1*x0'
                     },
      'submodels' : {'m6' : {'ref'    : 'M6', 
                             }
                     }
      }

m6 = {'submodels' : {'m7' : {'ref'    : 'M7', 
                             }
                     }
      }

m7 = {'submodels' : {'m5' : {'ref'    : 'M5', 
                             }
                     }
      }


all_data = {'M0': m0, 'M1': m1, 'M2': m2, 'M3': m3, 'M4': m4, 'M5': m5, 
            'M6': m6, 'M7': m7
            }