import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin            as dn
import dunlin.simulation as sim

###############################################################################
#Test Classes
###############################################################################
class TestSimulation:
    def test_1(self):
        plt.close('all')
        
        #Preprocessing
        model_data = dn.read_ini('_test/TestModel_1.ini')
        model      = model_data['model_1']['model']
        
        def obj1(t, y, p, u):
            s = y[0]
            
            mu_max = p[0]
            ks     = p[1]
            
            mu = mu_max*s/(s+ks)
            
            return t, mu
        
        def obj2(t, y, p, u):
            x = y[0]
            s = y[1]
            
            mu_max = p[0]
            ks     = p[1]
            ys     = p[2]
            
            mu = mu_max*s/(s+ks)
        
            dx = mu*x - 0.08*x
            
            return t, dx/ys
        
        def modify1(function, init, params, inputs, scenario, segment):
            new_init   = init.copy()
            new_params = params.copy()
            new_inputs = inputs.copy()
            
            new_init[0] *= 4
            
            return new_init, new_params, new_inputs
        
        #Test integration
        simulation_result = sim.integrate_model(model)
        scenario          = 0
        estimate          = 0
        table             = simulation_result[scenario][estimate][0]
        assert table.shape == (62, 9)
        
        #Test simulation with exv function
        model.exvs        = {1 : obj1, 2: obj2}
        simulation_result = sim.integrate_model(model)
        scenario          = 0
        estimate          = 0
        table, obj_vals   = simulation_result[scenario][estimate]
        model.exvs        = {}
        
        xo1, yo1 = obj_vals[1]
        xo2, yo2 = obj_vals[2]
        
        #Plot
        fig = plt.figure()
        AX  = [fig.add_subplot(5, 1, i+1) for i in range(5)]
        
        AX[0].plot(table['t'], table['x'])
        AX[1].plot(table['t'], table['s'])
        AX[2].plot(table['t'], table['b'])
        AX[3].plot(xo1, yo1)
        AX[4].plot(xo2, yo2)
        
        #Test modifier
        model.modify      = modify1
        model.exvs        = {1 : obj1, 2: obj2}
        simulation_result = sim.integrate_model(model)
        scenario          = 0
        estimate          = 0 
        table, obj_vals   = simulation_result[scenario][estimate]
        model.exvs        = {}
        
        xo1, yo1 = obj_vals[1]
        xo2, yo2 = obj_vals[2]
        
        assert xo1.shape == (62,)
        assert yo1.shape == (62,)
        assert xo2.shape == (62,)
        assert yo2.shape == (62,)
        model.modify = None
        
        #Plot
        fig = plt.figure()
        AX  = [fig.add_subplot(5, 1, i+1) for i in range(5)]
        
        AX[0].plot(table['t'], table['x'])
        AX[1].plot(table['t'], table['s'])
        AX[2].plot(table['t'], table['b'])
        AX[3].plot(xo1, yo1)
        AX[4].plot(xo2, yo2)
        
        #Test multi-model
        model.exvs = {1 : obj1, 2: obj2}
        sim_args  = {'model_1'    : {'model': model},
                     'model_2'    : {'model': model}
                     }
        
        simulation_results = sim.integrate_models(sim_args)
        model_key          = 'model_1'
        scenario           = 0
        estimate           = 0 
        table, obj_vals    = simulation_results[model_key][scenario][estimate]
        
        xo1, yo1 = obj_vals[1]
        xo2, yo2 = obj_vals[2]
        
        assert xo1.shape == (62,)
        assert yo1.shape == (62,)
        assert xo2.shape == (62,)
        assert yo2.shape == (62,)
        
        #Test plotting
        model.exvs = {1 : obj1, 2: obj2}
        sim_args   = {'model_1' : {'model' : model},
                      'model_2' : {'model' : model}
                      }
        
        simulation_results = sim.integrate_models(sim_args)
        
        #Test basic plot
        plot_index = {'model_1': ['x', 's', 'b', 1, 2]}
        figs, AX   = sim.plot_simulation_results(plot_index, simulation_results)
    
        assert len(AX['model_1']['x'].lines) == 4
        assert len(AX['model_1']['b'].lines) == 4
        assert len(AX['model_1'][  2].lines) == 4
        
        #Test line args
        plot_index = {'model_1': ['x', 'b', 1, 2],
                      'model_2': ['x', 'b', 1, 2]
                      }
        color      = {'model_1': {0: sim.colors['cobalt'],
                                  1: sim.colors['marigold']
                                  },
                      'model_2': sim.colors['teal']
                      }
        figs, AX   = sim.plot_simulation_results(plot_index, simulation_results, color=color, label='scenario')
        
        assert AX['model_1']['x'].lines[-1].get_label() == '_nolabel'
        
        #Test high-level
        model_data, sim_args = sim.read_ini('_test/TestModel_2.ini')
        plot_index           = {'model_1': ['x', 's', 'b', 'growth']}
        simulation_results   = sim.integrate_models(sim_args)
        figs, AX             = sim.plot_simulation_results(plot_index, simulation_results, color={'model_1': sim.colors['cobalt']})
    
        return

if __name__ == '__main__':
    T = TestSimulation()
    T.test_1()
    
    
