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
import dunlin                    as dn  
import dunlin.simulate           as sim
import dunlin.optimize.wrap_SSE  as ws
import dunlin.optimize.optimizer as opt
import dunlin.optimize.curvefit  as cf
import dunlin.utils_plot         as upp

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
      'optim_args': {'free_parameters' : {'u0': {'bounds': [0, 10], 
                                                 'prior': ['uniform', 0, 10]
                                                 },
                                          'u1': {'bounds': [0, 10], 
                                                 'prior': ['uniform', 0, 10]
                                                 },
                                          },
                     'settings'   : {'disp'   : False,
                                     'popsize': 5
                                     },
                     'line_args'  : {'color': {0: 'steel'}, 
                                     'marker': '+'
                                     }
                     }
      }

all_data = {'M1': m1}

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
    assert r == 1
    
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
    
    ###############################################################################
    #Part 2: OptResult Class
    ###############################################################################
    #Test on prior calculation on array
    print('Test on array')
    
    nominal      = {'p0': 50, 'p1': 50, 'p2': 50, 'p3': 50, 'p4': 50}
    nominal      = pd.DataFrame([nominal])
    free_params  = {'p1': {'bounds': [0,   100], 'scale': 'lin',   'prior': ['normal', 50, 10]},
                    'p3': {'bounds': [0.1, 100], 'scale': 'log10', 'prior': ['normal', 50, 10]}
                    }
    OptResult    = opt.Optimizer(nominal, free_params, lambda x: 0)
    free_p_array = np.array([1, 1])

    r  = OptResult.sampled_parameters[0](1)
    a0 = norm(50, 10).pdf(1)
    assert r == a0
    
    r  = OptResult.sampled_parameters[1](1)
    a1 = norm(50, 10).pdf(10)
    assert r == a1
    
    r = OptResult.get_objective(free_p_array)
    assert r == -logsum(np.array([a0 , a1]))
    
    free_params  = {'p1': {'bounds': [0,   100], 'scale': 'lin',   'prior': ['parameterScaleNormal', 50, 10]},
                    'p3': {'bounds': [0.1, 100], 'scale': 'log10', 'prior': ['parameterScaleNormal',  1,  1]}
                    }
    OptResult    = opt.Optimizer(nominal, free_params, lambda x: 0)
    free_p_array = np.array([1, 1])
    
    r  = OptResult.sampled_parameters[0](1)
    a0 = norm(50, 10).pdf(1)
    assert r == a0
    
    r  = OptResult.sampled_parameters[1](1)
    a1 = norm(1, 1).pdf(1)
    assert r == a1
    
    r = OptResult.get_objective(free_p_array)
    assert r == -logsum(np.array([a0 , a1]))
    
    #Test with log_likelihood
    print('Test with log_likelihood')
    OptResult.neg_log_likelihood = lambda x: sum(x)
    
    r    = OptResult.get_objective(free_p_array)
    temp = -logsum(np.array([a0 , a1]))
    temp = temp + sum(np.array([1, 10]))
    assert r == temp
    
    def log_likelihood(params):
        return sum([abs(params[0] - 50), abs(params[1]-10)])
    
    OptResult.neg_log_likelihood = log_likelihood
    
    r    = OptResult.get_objective(free_p_array)
    temp1 = sum([abs(free_p_array[0] - 50), abs(10**free_p_array[1]-10)])
    temp2 = -logsum(np.array([a0 , a1]))
    assert r == temp1 + temp2
    
    r    = OptResult.get_objective(np.array([50, 1]))
    temp1 = 0
    temp2 = -logsum(np.array([norm(50, 10).pdf(50) , norm(1, 1).pdf(1)]))
    assert r == temp1 + temp2
    
    ###############################################################################
    #Part 3: Optimize (Differential Evolution)
    ###############################################################################
    #Test differential evolution
    print('Test differential evolution')
    def log_likelihood(params):
        #Only the free params will be passed into this function
        return sum([abs(params[0] - 50), abs(params[1]-10)])
    
    nominal      = {'p0': 50, 'p1': 50, 'p2': 50, 'p3': 50, 'p4': 50}
    nominal      = pd.DataFrame([nominal])
    free_params  = {'p1': {'bounds': [0,   100], 'scale': 'lin',   'prior': ['parameterScaleNormal', 50, 10]},
                    'p3': {'bounds': [0.1, 100], 'scale': 'log10', 'prior': ['parameterScaleNormal',  1,  1]}
                    }
    optr = opt.Optimizer(nominal, free_params, log_likelihood)
    
    trace = optr.run_differential_evolution()
    
    o = trace.raw
    assert np.all( np.isclose(o.x, [50, 1], rtol=2e-2))
    
    a = trace.samples
    assert type(a) == pd.DataFrame

    ###############################################################################
    #Part 4: Test Curvefitting
    ###############################################################################
    #Read model
    model = dn.ODEModel('M1', **m1)
    
    #The y values are first order diff equations i.e. dx = g - p*x
    time    = np.linspace(0, 100, 51)
    y_data0 = 50 - 50*np.exp(-0.1*time)
    y_data1 = 50 + 50*np.exp(-0.1*time)
    
    #Formatted as scenario -> state -> series
    series0 = pd.Series(y_data0, index=time)
    series1 = pd.Series(y_data1, index=time)
    
    data = {0 : {'x0' : series0,
                 'x1' : series0
                 },
            1 : {'x0' : series1,
                 'x1' : series1
                 }
            }
    
    get_SSE = ws.SSECalculator(model, data=data, by='scenario')
    
    #Case 0: Test instantiation from model
    print('Test instantiation from model')
    optr = opt.Optimizer.from_model(model, to_minimize=get_SSE)
    
    trace = optr.run_differential_evolution()
    
    #Get the scipy result
    o = trace.raw
    assert all( np.isclose(o.x, [5, 5], rtol=1e-3) )
    
    ###########################################################################
    #Part 5: Plotting and Export with Trace
    ###########################################################################
    fig, AX = upp.figure(3, 2)
    
    
    trace.plot_steps(AX[0], 'objective')
    AX[0].set_title('Test plot_steps on objective')
    
    trace.plot_steps(AX[1], 'u0', 'u1')
    AX[1].set_title('Test plot_steps on parameters')
    
    trace.plot_kde(AX[2], 'u0')
    AX[2].set_title('Test plot_kde on 1 parameter')
    
    trace.plot_kde(AX[3], 'u0', 'u1')
    AX[3].set_title('Test plot_kde on 2 parameters')
    
    trace.plot_histogram(AX[4], 'u0')
    AX[4].set_title('Test plot_histogram on 1 parameter')
    
    trace.plot_histogram(AX[5], 'u0', 'u1')
    AX[5].set_title('Test plot_histogram on 2 parameters')
    
    fig.tight_layout()
    
    trace.to_excel('test_optimize.xlsx')