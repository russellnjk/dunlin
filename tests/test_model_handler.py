import numpy  as np
import pandas as pd

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
from dunlin import *
import dunlin._utils_model.integration as itg

###############################################################################
#Globals
###############################################################################
model = None

###############################################################################
#Test Classes
###############################################################################
class TestModelInstance:
    
    def test_1(self):
        #Read .ini
        #Case 1: Basic arguments
        model = read_ini('_test/TestModel_1.ini')['model_1']['model']
        assert model.states == ('x', 's', 'p')
        assert model.params == ('ks', 'mu_max', 'synp', 'ys')
        assert model.inputs == ('b',)
    
    
        #Case 2: With objective
        model = read_ini('_test/TestModel_2.ini')['model_1']['model']
        assert model.states == ('x', 's', 'p')
        assert model.params == ('ks', 'mu_max', 'synp', 'ys')
        assert model.inputs == ('b',)
        
        objectives = read_ini('_test/TestModel_2.ini')['model_1']['objectives']
        assert len(objectives) == 1
        
        #Test attributes related to integration   
        model    = read_ini('_test/TestModel_1.ini')['model_1']['model']
        
        tspan    = model.tspan
        assert len(tspan) == 2
        assert all( np.isclose(tspan[0], np.linspace(  0, 300, 31)) )
        assert all( np.isclose(tspan[1], np.linspace(300, 600, 31)) )
        
        init = model.init_vals
        y    = init.loc[0].values
        assert all(y == 1)
        
        params = model.param_vals
        p      = params.loc[0].values
        assert all(p == [20, 0.1, 1, 2])
        
        inputs = model.input_vals
        i      = inputs.loc[0].values
        i0     = i[0]
        assert all(i0 == [2])
        
        #Test model function
        t = 0
        f = model.func
        
        r = f(t, y, p, i0)
        assert all(r)
        
        #Test integration
        y_model, t_model = itg.piecewise_integrate(model.func, tspan, y, p, i, scenario=0)
        
        assert y_model.shape == (3, 62)
        
        #Test objective function
        objectives = read_ini('_test/TestModel_2.ini')['model_1']['objectives']
        obj_1      = objectives['growth']
        
        y_df      = pd.DataFrame([y, y+r], columns=model.states)
        t_df      = pd.DataFrame(t_model[:2][:,None], columns=['Time'])
        params_df = pd.DataFrame([p, p], columns=model.params)
        inputs_df = pd.DataFrame([i0, i0], columns=model.inputs)
        table     = pd.concat((t_df, y_df, params_df, inputs_df), axis=1)
        
        xo1, yo1 = obj_1(table)
        assert all(xo1 == t_model[:2])
        assert np.isclose(yo1[0], r[0])
    
if __name__ == '__main__':
    T = TestModelInstance()
    T.test_1()