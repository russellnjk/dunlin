import matplotlib.pyplot as     plt
import numpy             as     np
import pandas            as pd

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin                   as dn
import dunlin.optimize.wrap_SSE as ws

all_data = {'M1': {'states'     : {'x0': [1, 1], 
                                   'x1': [0, 0]
                                   },
                   'parameters' : {'p0': [0.1, 0.1], 
                                   'p1': [0.1, 0.1]
                                   },
                   'reactions'  : {'r0': {'stoichiometry' : {'x0' : -1}, 
                                          'rate'          : 'p0'
                                          }, 
                                   'r1': {'stoichiometry' : {'x1' : 1}, 
                                          'rate'          : 'p1'
                                                          }, 
                                   },
                   'variables'  : {'v0': 'x0'
                                   },
                   'int_args'   : {'method': 'LSODA'
                                   },
                   'optim_args' : {'free_parameters': {'p0' : {'bounds': [1e-1, 1e1]},
                                                       'p1' : {'bounds': [1e-1, 1e1]}
                                                       }
                                   }
                   },
            'M2': {'states'     : {'x0': [1, 1], 
                                   'x1': [0, 0]
                                  },
                   'parameters' : {'p0': [0.1, 0.1], 
                                   'p1': [0.1, 0.1]
                                   },
                   'reactions'  : {'r0': {'stoichiometry' : {'x0' : -1}, 
                                          'rate'          : 'p0'
                                          }, 
                                   'r1': {'stoichiometry' : {'x1' : 1}, 
                                          'rate'          : 'p1'
                                                          }, 
                                   },
                   'variables'  : {'v0': 'x0'
                                   },
                   'int_args'   : {'method': 'LSODA'
                                   },
                   'optim_args' : {'free_parameters': {'p0' : {'bounds': [1e-1, 1e1]},
                                                       'p1' : {'bounds': [1e-1, 1e1]}
                                                       }
                                   }
                   },
            }

def test_equivalence(d, d_, msg: str):
        for k, v in d.items():
            if type(v) == dict:
                for kk, vv in v.items():
                    try:
                        test = vv == d_[k][kk]
                        if hasattr(test, '__iter__'):
                            test = all(test)
                        if not test:
                            raise ValueError(f'Not equivalent. Keys {k} -> {kk}')
                    except Exception as e:
                        raise e
            else:
                try:
                    test = v == d_[k]
                    if hasattr(test, '__iter__'):
                        test = all(test)
                    if not test:
                        raise ValueError(f'Not equivalent. Key {k}')
                except Exception as e:
                    raise e
                
if __name__ == '__main__':
    time0   = np.linspace(0,  1, 21)
    time1   = np.linspace(0,  2, 21)
    y_data0 = np.e**(-time0)
    y_data1 = 2 -2*np.e**(-time1)

    # cols1 = pd.MultiIndex.from_product([['x0', 'x1'], [0]])
    # cols2 = pd.MultiIndex.from_product([['x0', 'x1'], [1]])
    # df1 = pd.DataFrame(np.array([y_data0, y_data0]).T, index=time0, columns=cols1)
    # df2 = pd.DataFrame(np.array([y_data1, y_data1]).T, index=time1, columns=cols2)
    
    s0 = pd.Series(y_data0, index=time0)
    s1 = pd.Series(y_data1, index=time1)
    
    s0.sd = 1
    s1.sd = 1
    
    by_state = {'x0': {0: s0,
                       1: s1,
                       },
                'x1': {0: s0,
                       1: s1,
                       },
                } 
    
    by_scenario = {0: {'x0': s0,
                       'x1': s0,
                       },
                   1: {'x0': s1,
                       'x1': s1
                       },
                   }
    
    model = dn.ODEModel.from_data(all_data, 'M1')
    
    ###########################################################################
    #Test Preprocessing
    ###########################################################################
    r = ws.SSECalculator.parse_data(model, by_scenario, by='scenario')
    
    scenario2y_data0  = r[0]
    scenario2t_data0  = r[1]
    scenario2sd_data0 = r[2] 
    scenario2t_idxs0  = r[3] 
    scenario2tspan0   = r[4]
    
    r = ws.SSECalculator.parse_data(model, by_state, by='state')
    
    scenario2y_data1  = r[0]
    scenario2t_data1  = r[1]
    scenario2sd_data1 = r[2] 
    scenario2t_idxs1  = r[3] 
    scenario2tspan1   = r[4]
    
    test_equivalence(scenario2y_data0,  scenario2y_data1,  'scenario2y_data' )
    test_equivalence(scenario2t_data0,  scenario2t_data1,  'scenario2t_data' )
    test_equivalence(scenario2sd_data0, scenario2sd_data1, 'scenario2sd_data')
    test_equivalence(scenario2t_idxs0,  scenario2t_idxs1,  'scenario2t_idxs' )
    test_equivalence(scenario2tspan0,   scenario2tspan1,   'scenario2tspan'  )
    
    assert set(scenario2y_data0)  == {0, 1}
    assert set(scenario2t_data0)  == {0, 1}
    assert set(scenario2sd_data0) == {0, 1}
    assert set(scenario2t_idxs0)  == {0, 1}
    assert set(scenario2tspan0)   == {0, 1}
    
    ###########################################################################
    #Test Instantiation
    ###########################################################################
    get_SSE = ws.SSECalculator(model, by_scenario, by='scenario')
    
    ###########################################################################
    #Test Calculation
    ###########################################################################    
    get_SSE = ws.SSECalculator(model, by_scenario, by='scenario')
    
    #Initialize variables
    init = model.state_dict
    assert np.all( init[0] == np.array([1, 0]) )
    assert np.all( init[1] == np.array([1, 0]) )
    
    #Test 0
    print('Test wrap_get_SSE 0')
    p_array  = np.array([1, 1])
    SSE      = get_SSE(p_array)
    print(SSE)
    assert np.isclose(59.348280344968195, SSE, rtol=1e-2)
    
    #Test 1
    delattr(s0, 'sd')
    delattr(s1, 'sd')
    
    get_SSE = ws.SSECalculator(model, by_scenario, by='scenario')
    
    print('Test wrap_get_SSE 0')
    p_array  = np.array([1, 1])
    SSE      = get_SSE(p_array)
    print(SSE)
    assert np.isclose(842.5211496180012, SSE, rtol=1e-2)
    
    