import matplotlib.pyplot as     plt
import numpy             as     np
import pandas            as     pd
from numba               import njit

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin                          as dn  
import dunlin.model                    as dml
import dunlin.simulation               as sim
import dunlin._utils_optimize.wrap_SSE as ws

if __name__ == '__main__':
    time1   = np.linspace(0,  1, 21)
    time2   = np.linspace(0,  2, 21)
    y_data1 = np.e**(-time1)
    y_data2 = 2 -2*np.e**(-time2)
    dataset = {('Data', 0, 'x0') : y_data1,
               ('Time', 0, 'x0') : time1,
               ('Data', 0, 'x1') : y_data1,
               ('Time', 0, 'x1') : time1,
               ('Data', 1, 'x0') : y_data2,
               ('Time', 1, 'x0') : time2,
               ('Data', 1, 'x1') : y_data2,
               ('Time', 1, 'x1') : time2,               
               }
    
    #Read model
    dun_data, models = dn.read_file('optimization_test_files/SSE1.dun')
    model            = models['M1']
    
    #Test Extraction of tspan, indices for t, y values and st dev
    print('Test Extraction of tspan, indices for t, y values and st dev')
    tspan, t_data, y_data, s_data, exv_names = ws.split_dataset(model, dataset)
    
    #Initialize variables
    init = ws.get_init(model)
    assert np.all( init[0] == np.array([1, 0]) )
    assert np.all( init[1] == np.array([1, 0]) )
    
    print('Test wrap_get_SSE')
    p_array  = np.array([1, 1])
    get_SSE  = ws.wrap_get_SSE(model, dataset)
    SSE      = get_SSE(p_array)
    print(SSE)
    assert np.isclose(953.4656701575371, SSE, rtol=1e-2)
    
    #Test mismatched data/time length
    print('Test mismatched data/time length')
    dataset1 = dict(dataset)
    dataset[('Time', 1, 'x1')] = [*time2, 2.1]
    try:
        tspan, t_data, y_data, s_data = ws.split_dataset(model, dataset)
    except ValueError:
        assert True
    else:
        assert False
    
    #Test missing exv
    print('Test mismatched data/time length')
    dataset1 = dict(dataset)
    dataset[('Data', 1, 'xx1')] = np.array([1, 2, 3])
    try:
        tspan, t_data, y_data, s_data = ws.split_dataset(model, dataset)
    except ValueError:
        assert True
    else:
        assert False
        
    #Read model
    dun_data, models = dn.read_file('optimization_test_files/SSE2.dun')
    model            = models['M1']
    
    print('Test wrap_get_SSE with exv')
    dataset  = {('Data', 0, 'exv0') : y_data1,
                ('Time', 0, 'exv0') : time1,
                ('Data', 1, 'exv0') : y_data2,
                ('Time', 1, 'exv0') : time2,            
                }
    p_array  = np.array([1, 1])
    get_SSE  = ws.wrap_get_SSE(model, dataset)
    SSE      = get_SSE(p_array)
    
    dataset  = {('Data', 0, 'x0') : y_data1,
                ('Time', 0, 'x0') : time1,
                ('Data', 1, 'x0') : y_data2,
                ('Time', 1, 'x0') : time2,            
                }
    SSE_     = get_SSE(p_array)
    assert np.isclose(SSE, SSE_, rtol=1e-4)

    
    
    
    
    
    
    
    
    