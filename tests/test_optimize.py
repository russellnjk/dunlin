import matplotlib.pyplot as     plt
import numpy             as     np
import pandas            as     pd
import scipy.optimize    as     sop
from numba               import njit
from scipy.stats         import norm, laplace, lognorm, loglaplace, uniform

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin                          as dn  
import dunlin.model                    as dml
import dunlin.simulate                 as sim
import dunlin._utils_optimize.wrap_SSE as ws
import dunlin.optimize                 as opt

@njit
def logsum(arr):
    return np.sum(np.log(arr))
    
if __name__ == '__main__':
    plt.close('all')
    
    ###############################################################################
    #Part 1: Sampled Params Class
    ###############################################################################
    free_params = {'p0': {'bounds': [0, 10], 'scale': 'log10', 'prior': ['uniform', 0, 10]},
                    'p1': {'bounds': [0, 10], 'scale': 'log',   'prior': ['uniform', 0, 10]},
                    'p2': {'bounds': [0, 10], 'scale': 'lin',   'prior': ['uniform', 0, 10]},
                    }
    #Test scaling
    print('Test scaling')
    p0 = opt.SampledParam('p0', **free_params['p0'])
    
    assert p0.scale(10)  == 1
    assert p0.unscale(1) == 10
    
    p1 = opt.SampledParam('p1', **free_params['p1'])
    
    assert p1.scale(np.e) == 1
    assert p1.unscale(1)  == np.e
    
    p2 = opt.SampledParam('p2', **free_params['p2'])
    
    assert p2.scale(3)   == 3
    assert p2.unscale(3) == 3
    
    #Test prior calculation for a single parameter
    print('Test prior calculation for a single parameter')
    
    free_params = {'p0': {'bounds': [-10, 10], 'scale': 'lin',   'prior': ['uniform',              -5, 5]},
                    'p1': {'bounds': [-10, 10], 'scale': 'lin',   'prior': ['normal',                0, 1]},
                    'p2': {'bounds': [-10, 10], 'scale': 'lin',   'prior': ['laplace',               0, 1]},
                    'p3': {'bounds': [0.1, 10], 'scale': 'log10', 'prior': ['normal',                0, 1]},
                    'p4': {'bounds': [0.1, 10], 'scale': 'log10', 'prior': ['parameterScaleNormal',  0, 1]},
                    'p5': {'bounds': [0.1, 10], 'scale': 'log10', 'prior': ['normal',                0, 1], 
                          'sample': ['laplace', 0, 1]},
                    }
    
    p0 = opt.SampledParam('p0', **free_params['p0'])
    r = p0.get_prior(1)    
    assert r == uniform(0, 10).pdf(1)
    
    p1 = opt.SampledParam('p1', **free_params['p1'])
    r = p1.get_prior(1)    
    assert r == norm(0, 1).pdf(1)
    
    p2 = opt.SampledParam('p2', **free_params['p2'])
    r = p2.get_prior(1)    
    assert r == laplace(0, 1).pdf(1)
    
    p3 = opt.SampledParam('p3', **free_params['p3'])
    r = p3.get_prior(1)    
    assert r == norm(0, 1).pdf(10)
    
    p4 = opt.SampledParam('p4', **free_params['p4'])
    r = p4.get_prior(1)    
    assert r == norm(0, 1).pdf(1)
    
    p5 = opt.SampledParam('p5', **free_params['p5'])
    r = p5.get_prior(1)    
    assert r == norm(0, 1).pdf(10)
    
    # ###############################################################################
    # #Part 2: OptResult Class
    # ###############################################################################
    # #Test on prior calculation on array
    # print('Test on array')
    
    # nominal      = {'p0': 50, 'p1': 50, 'p2': 50, 'p3': 50, 'p4': 50}
    # free_params  = {'p1': {'bounds': [0,   100], 'scale': 'lin',   'prior': ['normal', 50, 10]},
    #                 'p3': {'bounds': [0.1, 100], 'scale': 'log10', 'prior': ['normal', 50, 10]}
    #                 }
    # OptResult    = opt.OptResult(nominal, free_params, lambda x: 0)
    # free_p_array = np.array([1, 1])

    # r  = OptResult.sampled_params[0](1)
    # a0 = norm(50, 10).pdf(1)
    # assert r == a0
    
    # r  = OptResult.sampled_params[1](1)
    # a1 = norm(50, 10).pdf(10)
    # assert r == a1
    
    # r = OptResult.get_objective(free_p_array)
    # assert r == -logsum(np.array([a0 , a1]))
    
    # free_params  = {'p1': {'bounds': [0,   100], 'scale': 'lin',   'prior': ['parameterScaleNormal', 50, 10]},
    #                 'p3': {'bounds': [0.1, 100], 'scale': 'log10', 'prior': ['parameterScaleNormal',  1,  1]}
    #                 }
    # OptResult    = opt.OptResult(nominal, free_params, lambda x: 0)
    # free_p_array = np.array([1, 1])
    
    # r  = OptResult.sampled_params[0](1)
    # a0 = norm(50, 10).pdf(1)
    # assert r == a0
    
    # r  = OptResult.sampled_params[1](1)
    # a1 = norm(1, 1).pdf(1)
    # assert r == a1
    
    # r = OptResult.get_objective(free_p_array)
    # assert r == -logsum(np.array([a0 , a1]))
    
    # #Test with log_likelihood
    # print('Test with log_likelihood')
    # OptResult.neg_log_likelihood = lambda x: sum(x)
    
    # r    = OptResult.get_objective(free_p_array)
    # temp = -logsum(np.array([a0 , a1]))
    # temp = temp + sum(np.array([1, 10]))
    # assert r == temp
    
    # def log_likelihood(params):
    #     return sum([abs(params[0] - 50), abs(params[1]-10)])
    
    # OptResult.neg_log_likelihood = log_likelihood
    
    # r    = OptResult.get_objective(free_p_array)
    # temp1 = sum([abs(free_p_array[0] - 50), abs(10**free_p_array[1]-10)])
    # temp2 = -logsum(np.array([a0 , a1]))
    # assert r == temp1 + temp2
    
    # r    = OptResult.get_objective(np.array([50, 1]))
    # temp1 = 0
    # temp2 = -logsum(np.array([norm(50, 10).pdf(50) , norm(1, 1).pdf(1)]))
    # assert r == temp1 + temp2
    
    # ###############################################################################
    # #Part 3: optimize (Differential Evolution)
    # ###############################################################################
    # #Test differential evolution
    # print('Test differential evolution')
    # def log_likelihood(params):
    #     #Only the free params will be passed into this function
    #     return sum([abs(params[0] - 50), abs(params[1]-10)])
    
    # nominal      = {'p0': 50, 'p1': 50, 'p2': 50, 'p3': 50, 'p4': 50}
    # free_params  = {'p1': {'bounds': [0,   100], 'scale': 'lin',   'prior': ['parameterScaleNormal', 50, 10]},
    #                 'p3': {'bounds': [0.1, 100], 'scale': 'log10', 'prior': ['parameterScaleNormal',  1,  1]}
    #                 }
    # OptResult    = opt.OptResult(nominal, free_params, log_likelihood)
    
    # r = OptResult.differential_evolution()
    
    # o = r['o']
    # assert np.all( np.isclose(o.x, [50, 1], rtol=2e-2))
    
    # a = r['a']
    # assert type(a) == pd.DataFrame

    # ###############################################################################
    # #Part 4A: Test Curvefitting
    # ###############################################################################
    # #Read model
    # dun_data, models = dn.read_file('optimize_test_files/differential_evolution1.dun')
    
    # model       = models['M1']
    # time        = np.linspace(0, 100, 51)
    # y_data0     = 50 - 50*np.exp(-0.1*time)
    # y_data1     = 50 + 50*np.exp(-0.1*time)
    # dataset     = {('Data', 0, 'x0') : y_data0,
    #                ('Time', 0, 'x0') : time,
    #                ('Data', 0, 'x1') : y_data0,
    #                ('Time', 0, 'x1') : time,
    #                ('Data', 1, 'x0') : y_data1,
    #                ('Time', 1, 'x0') : time,
    #                ('Data', 1, 'x1') : y_data1,
    #                ('Time', 1, 'x1') : time,               
    #                }
    # free_params = {'u0': {'bounds': [0, 10], 'prior': ['uniform', 0, 10]},
    #                'u1': {'bounds': [0, 10], 'prior': ['uniform', 0, 10]},
    #                }
    
    # model.optim_args = {'free_params': free_params}
    # get_SSE          = ws.SSECalculator(model, dataset)
    # fig, AX          = dn.figure(1, 1)
    
    # #Case 0: Test instantiation from model
    # print('Test instantiation from model')
    # model.optim_args = {'free_params': free_params,
    #                     'settings'   : {'disp'   : False,
    #                                     'popsize': 5
    #                                     }
    #                     }
    # optresult = opt.OptResult.from_model(model, to_minimize=get_SSE)
    
    # r = optresult.differential_evolution()
    
    # o = r['o']
    # assert all( np.isclose(o.x, [5, 5], rtol=1e-3) )
    # AX[0].plot(r['p'], label='Case 1')
    # AX[0].legend()
    
    # #Case 1: Function-based parameter estimation
    # print('Function-based parameter estimation')
    # opt_results = opt.fit_model(model, dataset, algo='differential_evolution')
    # r           = opt_results[0]
    
    # o = r.o
    # assert all( np.isclose(o.x, [5, 5], rtol=1e-3) )
    # assert type(r.a) == pd.DataFrame
    # AX[0].plot(r.posterior, label='Case 3')
    # AX[0].legend()
    
    # ###############################################################################
    # #Part 4B: Test EXV optimize
    # ###############################################################################
    
    # ###############################################################################
    # #Part 5: Visualization
    # ###############################################################################
    # dun_data, models = dn.read_file('optimize_test_files/differential_evolution1.dun')
    
    # model       = models['M1']
    # time        = np.linspace(0, 100, 51)
    # y_data0     = 50 - 50*np.exp(-0.1*time)
    # y_data1     = 50 + 50*np.exp(-0.1*time)
    # dataset     = {('Data', 0, 'x0') : y_data0,
    #                 ('Time', 0, 'x0') : time,
    #                 ('Data', 0, 'x1') : y_data0,
    #                 ('Time', 0, 'x1') : time,
    #                 ('Data', 1, 'x0') : y_data1,
    #                 ('Time', 1, 'x0') : time,
    #                 ('Data', 1, 'x1') : y_data1,
    #                 ('Time', 1, 'x1') : time,               
    #                 }
    
    # free_params = {'u0': {'bounds': [0, 10], 'prior': ['uniform', 0, 10]},
    #                'u1': {'bounds': [0, 10], 'prior': ['uniform', 0, 10]},
    #                }
    # fig, AX     = dn.figure(2, 1)
    # AX_         = {'u0': AX[0], 'u1': AX[1]}
    
    # model.optim_args = {'free_params': free_params,
    #                     'settings'   : {'disp'   : True,
    #                                     'popsize': 5
    #                                     }, 
    #                     'line_args'  : {'color': {0: 'steel'}, 
    #                                     'marker': '+'
    #                                     }
    #                     }
    # get_SSE          = ws.SSECalculator(model, dataset)
    
    # print('Testing visualization')
    # opt_results = opt.fit_model(model, dataset, algo='differential_evolution')
    # AX_         = opt.plot_traces(opt_results, AX_)
    
    # ###############################################################################
    # #Part 6: .dun File
    # ###############################################################################
    # dun_data, models = dn.read_file('optimize_test_files/differential_evolution2.dun')
    
    # model       = models['M1']
    # time        = np.linspace(0, 100, 51)
    # y_data0     = 50 - 50*np.exp(-0.1*time)
    # y_data1     = 50 + 50*np.exp(-0.1*time)
    # dataset     = {('Data', 0, 'x0') : y_data0,
    #                 ('Time', 0, 'x0') : time,
    #                 ('Data', 0, 'x1') : y_data0,
    #                 ('Time', 0, 'x1') : time,
    #                 ('Data', 1, 'x0') : y_data1,
    #                 ('Time', 1, 'x0') : time,
    #                 ('Data', 1, 'x1') : y_data1,
    #                 ('Time', 1, 'x1') : time,               
    #                 }
    
    # # print('Testing .dun file')
    # # fig, AX     = dn.figure(2, 1)
    # # AX_         = {'u0': AX[0], 'u1': AX[1]}
    # # opt_results = opt.fit_model(model, dataset, 'differential_evolution')
    # # AX_         = opt.plot_opt_results(opt_results, AX_)
    
    # # fig, AX     = dn.figure(1, 1)
    # # AX_         = {('u0', 'u1'): AX[0]}
    # # AX_         = opt.plot_opt_results(opt_results, AX_)
    
    # print('Test exv parameter estimation.')
    # fig, AX     = dn.figure(2, 1)
    # AX_         = {'u0': AX[0], 'u1': AX[1]}
    # dataset     = {('Data', 0, 'exv0') : y_data0,
    #                ('Time', 0, 'exv0') : time,
    #                ('Data', 0, 'exv1') : y_data0,
    #                ('Time', 0, 'exv1') : time,
    #                ('Data', 1, 'exv0') : y_data1,
    #                ('Time', 1, 'exv0') : time,
    #                ('Data', 1, 'exv1') : y_data1,
    #                ('Time', 1, 'exv1') : time,               
    #                }
    # opt_results = opt.fit_model(model, dataset, algo='differential_evolution')
    # AX_         = opt.plot_traces(opt_results, AX_)
    
    # fig2, AX2     = dn.figure(1, 1)
    # AX2_         = {('u0', 'u1'): AX2[0]}
    # AX2_         = opt.plot_traces(opt_results, AX2_)
    
    # #Try a different exv
    # AX_         = {'u0': AX[0], 'u1': AX[1]}
    # dataset     = {('Data', 0, 'exv2') : np.array([0, 0]),
    #                ('Time', 0, 'exv2') : np.array([0, 100]),  
    #                ('Std',  0, 'exv2') : 1
    #                }
    # opt_results = opt.fit_model(model, dataset, algo='differential_evolution')
    # AX_         = opt.plot_traces(opt_results, AX_, color='ocean')
    # AX2_        = opt.plot_traces(opt_results, AX2_, color='ocean')
    
    
    