import matplotlib.pyplot as     plt
import numpy             as     np
import pandas            as pd

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin                   as dn
import dunlin.optimize.wrap_SSE as ws

plt.close('all')
plt.ion()

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
                   'opt_args'   : {'free_parameters': {'p0' : {'bounds': [1e-1, 1e1]},
                                                       'p1' : {'bounds': [1e-1, 1e1]}
                                                       }
                                   },
                   'data_args'  : {'line_args' : {'color' : {0: 'cobalt',
                                                             1: 'tangerine'
                                                             },
                                                  'marker'     : 'o', 
                                                  'markersize' : 5, 
                                                  'linestyle'  : 'None'
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
                   'opt_args'   : {'free_parameters': {'p0' : {'bounds': [1e-1, 1e1]},
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

    s0 = pd.Series(y_data0, index=time0)
    s1 = pd.Series(y_data1, index=time1)
    
    s0.sd = 1
    s1.sd = 1
    
    data = {('x0', 0) : s0,
            ('x0', 1) : s1,
            ('x1', 0) : s0,
            ('x1', 1) : s1
            }
    
    model = dn.ODEModel.from_data(all_data, 'M1')
    
    ###########################################################################
    #Test Preprocessing
    ###########################################################################
    r = ws.SSECalculator.parse_data(model, data)
    
    scenario2y_data0  = r[0]
    scenario2t_data0  = r[1]
    scenario2sd_data0 = r[2] 
    scenario2t_idxs0  = r[3] 
    scenario2tspan0   = r[4]
    
    assert set(scenario2y_data0)  == {0, 1}
    assert set(scenario2t_data0)  == {0, 1}
    assert set(scenario2sd_data0) == {0, 1}
    assert set(scenario2t_idxs0)  == {0, 1}
    assert set(scenario2tspan0)   == {0, 1}
    
    ###########################################################################
    #Test Instantiation
    ###########################################################################
    get_SSE = ws.SSECalculator(model, data)
    
    ###########################################################################
    #Test Calculation
    ###########################################################################    
    get_SSE = ws.SSECalculator(model, data)
    
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
    
    get_SSE = ws.SSECalculator(model, data)
    
    print('Test wrap_get_SSE 0')
    p_array  = np.array([1, 1])
    SSE      = get_SSE(p_array)
    print(SSE)
    assert np.isclose(842.5211496180012, SSE, rtol=1e-2)
    
    ###########################################################################
    #Test Multiple Trials(Replicates)
    ###########################################################################    
    s2 = s0 + 1
    s3 = pd.concat({'a': s0, 'b': s2})
    s3.index.names = ['trial', 'time']

    data = {('x0', 0) : s3,
            ('x0', 1) : s1,
            ('x1', 0) : s3,
            ('x1', 1) : s1
            }
    
    get_SSE = ws.SSECalculator(model, data)
    
    #Test 0
    print('Test wrap_get_SSE with multiple trials(replicates)')
    gb = s3.groupby('time')
    
    assert all(get_SSE.scenario2sd_data[0]['x0'] == gb.std())
    assert all(get_SSE.scenario2y_data[ 0]['x0'] == gb.mean())
    
    p_array  = np.array([1, 1])
    SSE      = get_SSE(p_array)
    print(SSE)
    assert np.isclose(716.4965748640595, SSE, rtol=1e-2)
    
    ###########################################################################
    #Test Plotting
    ###########################################################################
    print('Test plotting')
    fig, AX = plt.subplots(1, 2)
    
    s0.sd = 0.1
    s1.sd = 0.2
    
    data = {('x0', 0) : s0,
            ('x0', 1) : s1,
            ('x1', 0) : s0,
            ('x1', 1) : s1
            }
    
    get_SSE = ws.SSECalculator(model, data)
    
    get_SSE.plot_data(AX[0], 'x0')
    get_SSE.plot_data(AX[1], 
                      'x1', 
                      colors         = {0: ['cobalt', 'light blue'], 
                                        1: ['tangerine']
                                        },
                      ignore_default = True
                      )
    
    print()
    print('Test plotting with multiindex')
    fig, AX = plt.subplots(1, 2)
    data = {('x0', 0) : s3,
            ('x0', 1) : s1,
            ('x1', 0) : s3,
            ('x1', 1) : s1
            }
    
    get_SSE = ws.SSECalculator(model, data)
    
    get_SSE.plot_data(AX[0], 'x0')
    get_SSE.plot_data(AX[1], 
                      'x1', 
                      colors         = {0: ['cobalt', 'light blue'], 
                                        1: ['tangerine']
                                        },
                      ignore_default = True
                      )
    
    ###########################################################################
    #Test Access
    ###########################################################################
    assert get_SSE.contains_var('x0') == True 
    assert get_SSE.contains_var(('x0', 'x1')) == True 
    
    