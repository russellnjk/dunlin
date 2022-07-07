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
                   'reactions'  : {'r0': ['x0 ->', 'p0'], 
                                   'r1': ['-> x1', 'p1']
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
                   'reactions'  : {'r0': ['x0 ->', 'p0'], 
                                               'r1': ['-> x1', 'p1']
                                               },
                   'variables'  : {'v0': 'x0'
                                   },
                   'extra'      : {'final_x0': ['index', 'x0', -1]
                                   },
                   'int_args'   : {'method': 'LSODA'
                                   },
                   'optim_args' : {'free_parameters': {'p0' : {'bounds': [1e-1, 1e1]},
                                                       'p1' : {'bounds': [1e-1, 1e1]}
                                                       }
                                   }
                   },
            }

if __name__ == '__main__':
    time1   = np.linspace(0,  1, 21)
    time2   = np.linspace(0,  2, 21)
    y_data1 = np.e**(-time1)
    y_data2 = 2 -2*np.e**(-time2)

    cols1 = pd.MultiIndex.from_product([['x0', 'x1'], [0]])
    cols2 = pd.MultiIndex.from_product([['x0', 'x1'], [1]])
    df1 = pd.DataFrame(np.array([y_data1, y_data1]).T, index=time1, columns=cols1)
    df2 = pd.DataFrame(np.array([y_data2, y_data2]).T, index=time2, columns=cols2)
    
    #Read model
    model    = dn.ODEModel.from_data(all_data, 'M1')

    #Test Extraction of tspan, indices for t, y values and st dev
    print('Test Extraction of tspan, indices for t, y values and st dev')
    tspan, y_data, s_data, t_data, t_idx = ws.SSECalculator.parse_df(model, df1, df2)
    
    #Initialize variables
    init = model._states
    assert np.all( init[0] == np.array([1, 0]) )
    assert np.all( init[1] == np.array([1, 0]) )
    
    print('Test wrap_get_SSE')
    df1_sd = df1.copy()
    df1_sd.columns = df1_sd.columns.set_levels(['__x0', '__x1'], level=0)
    df2_sd = df2.copy()
    df2_sd.columns = df2_sd.columns.set_levels(['__x0', '__x1'], level=0)
    for df in [df1_sd, df2_sd]:
        for col in df.columns:
            df[col].values[:] = 1
    
    p_array  = np.array([1, 1])
    get_SSE  = ws.SSECalculator(model, df1, df2, df1_sd, df2_sd)
    SSE      = get_SSE(p_array)
    print(SSE)
    assert np.isclose(59.348280344968195, SSE, rtol=1e-2)
        
    #Test variable SSE
    print('Test wrap_get_SSE with variable')
    p_array  = np.array([1, 1])
    get_SSE  = ws.SSECalculator(model, df1, df2)
    SSE      = get_SSE(p_array)
    

    SSE_     = get_SSE(p_array)
    assert np.isclose(SSE, SSE_, rtol=1e-4)

    
    
    
    
    
    
    
    
    