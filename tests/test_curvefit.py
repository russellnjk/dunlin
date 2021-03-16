import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin.model_handler as mh
import dunlin.simulation    as sim
import dunlin.curvefit      as cf

###############################################################################
#Test Classes
###############################################################################
class TestCurveFit:
    def test_param_check_1(self):
        #Test missing param check
        model_params = ['a', 'b']
        other_params = ['a']
        
        try:
            cf.check_missing_params(model_params, other_params, name='guess')
            assert False
        except cf.ParameterError:
            assert True
        except cf.AssertionError as e:
            raise e
        except Exception as e:
            raise e
        
        try:
            cf.check_missing_params(other_params, model_params, name='guess')
            assert False
        except cf.AssertionError as e:
            assert True
        except cf.Exception as e:
            raise e
        
        #Test unexpected param check
        model_params  = ['a', 'b']
        other_params  = ['a']
        other_params_ = ['c']
        
        try:
            cf.check_unexpected_params(model_params, other_params_, name='guess')
            assert False
        except cf.ParameterError:
            assert True
        except cf.AssertionError as e:
            raise e
        except Exception as e:
            raise e
        
        try:
            cf.check_unexpected_params(model_params, other_params, name='guess')
            assert False
        except AssertionError as e:
            assert True
        except Exception as e:
            raise e
    
    def test_simulated_annealing_1(self):
        plt.close('all')
        
        #Test simulated annealing 
        #Make data
        time1   = np.linspace(0,  1, 21)
        time2   = np.linspace(0,  2, 21)
        y_data1 = np.e**(-np.linspace(0, 1, 21))
        y_data2 = 2 -2*np.e**(-np.linspace(0, 2, 21))
        dataset = {('x', 0, 'Data') : y_data1,
                   ('x', 0, 'Time') : time1,
                   ('x', 1, 'Data') : y_data2,
                   ('x', 1, 'Time') : time2,
                   ('w', 0, 'Data') : y_data1,
                   ('w', 0, 'Time') : time1,
                   ('w', 1, 'Data') : y_data2,
                   ('w', 1, 'Time') : time2,               
                   }
        exp_data = {'model_1': dataset,
                    'model_2': dataset
                    }
        
        #Read model
        model_data = mh.read_ini('_test/TestCurveFit_3.ini')
        models     = {key: value['model'] for key, value in model_data.items()}
        
        #Make guess
        guess        = {'a' : 10, 'b': 10, 'c': 2, 'e': 10, 'f': 2}
        step_size    = {'a': 1, 'b': 1, 'e': 2}
        
        #SSE only
        opt_result   = cf.simulated_annealing(models, exp_data, guess, step_size, iterations=1000)    
        accepted     = opt_result['accepted']
        best_row     = accepted.iloc[np.argmax(opt_result['values'])].values
        objective    = opt_result['posterior']
        assert np.isclose(objective(best_row), 0, atol=2)
        
        #With priors
        priors       = {'a': np.array([8, 0.1]), 'b': np.array([8, 0.1])}
        opt_result   = cf.simulated_annealing(models, exp_data, guess, step_size, iterations=1000, priors=priors)    
        accepted     = opt_result['accepted']
        best_row     = accepted.iloc[np.argmax(opt_result['values'])].values
        objective    = opt_result['posterior']
        assert all(np.isclose(best_row[:2], 8, atol=0.5))
        
        #With priors and bounds
        bounds       = {'a': np.array([8.5, 12]), 'b': np.array([8.5, 12])}
        priors       = {'a': np.array([8, 0.1]),  'b': np.array([8, 0.1]) }
        opt_result   = cf.simulated_annealing(models, exp_data, guess, step_size, iterations=1000, priors=priors, bounds=bounds)    
        accepted     = opt_result['accepted']
        best_row     = accepted.iloc[np.argmax(opt_result['values'])].values
        objective    = opt_result['posterior']
        assert all(np.isclose(best_row[:2], 8.5, atol=0.5))
        assert all(accepted['a'] >= 8.5)
        assert all(accepted['b'] >= 8.5)
    
        #Test reading from .ini
        #Read model
        model_data = mh.read_ini('_test/TestCurveFit_4.ini')
        
        #Test argument extraction
        guesses, cf_args = cf.get_sa_args(model_data)
        
        models     = cf_args['models']
        priors     = cf_args['priors']
        bounds     = cf_args['bounds']
        iterations = cf_args['iterations']
        step_size  = cf_args['step_size']
        
        assert priors['a']    == [8,   0.1]
        assert priors['b']    == [8,   0.1]
        assert bounds['a']    == [8.5, 12 ]
        assert bounds['b']    == [8.5, 12 ]
        assert bounds['e']    == [1,   12 ]
        assert iterations     == 1000
        assert step_size['a'] == 1
        assert step_size['b'] == 1
        assert step_size['e'] == 2
        assert all([key in guesses[0] for key in ['a', 'b', 'c', 'e', 'f']])
        
        cf_args['exp_data'] = exp_data
        
        #Direct insertion of arguments
        opt_result   = cf.simulated_annealing(guess=guesses[0], **cf_args)    
        accepted     = opt_result['accepted']
        best_row     = accepted.iloc[np.argmax(opt_result['values'])].values
        objective    = opt_result['posterior']
        assert all(np.isclose(best_row[:2], 8.5, atol=0.3))
        assert all(accepted['a'] >= 8.5)
        assert all(accepted['b'] >= 8.5)
        
        #Visual check
        plt.close('all')
        fig = plt.figure()
        ax  = fig.add_subplot(1, 1, 1)
        ax.plot(accepted['a'], accepted['b'], marker='o', markersize=4)
        
        #Test plotting
        #Test plotting posterior
        posterior = pd.DataFrame.from_dict({'a': [0.5, 1],
                                            'b': [0.1, 1],
                                            'c': [2,   2],
                                            'e': [1,   1],
                                            'f': [2,   2]
                                            })
        
        plot_index = {'model_1': ['x', 'w'],
                      'model_2': ['x', 'w'],
                      }
        
        colors   = {'model_1': {0 : cf.colors['cobalt'],
                                1 : cf.colors['coral'],
                                },
                    'model_2': cf.colors['marigold']
                    } 
        
        sim_args         = sim.get_sim_args(model_data)
        guesses, cf_args = cf.get_sa_args(model_data)
        
        #Test plotting posterior
        simulation_results = cf.integrate_posteriors(sim_args, posterior)
        figs, AX           = cf.plot_posterior(plot_index, simulation_results)
        
        #Test plotting exp data
        figs, AX = cf.plot_exp_data(plot_index, exp_data)
        
        #Test high-level
        figs, AX, _, _ = cf.integrate_and_plot(plot_index = plot_index, 
                                               sim_args   = sim_args, 
                                               posterior  = posterior, 
                                               guesses    = guesses, 
                                               exp_data   = exp_data
                                               )
        return

if __name__ == '__main__':
    T = TestCurveFit()
    T.test_simulated_annealing_1()
    
    print('mh',       mh.Model,       id(mh.Model))
    print('cf.mh',    cf.mh.Model,    id(cf.mh.Model))
    print('cf.ws.mh', cf.ws.mh.Model, id(cf.ws.mh.Model))