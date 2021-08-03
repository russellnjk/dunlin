import numpy  as np
import pandas as pd

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin as dn

###############################################################################
#Test Classes
###############################################################################
class TestModelInstance:
    
    def test_read_1(self):
        #Read .ini
        model = dn.read_ini('_test/TestModel_1.ini')['model_1']['model']
        assert model.states == ('x', 's', 'h')
        assert model.params == ('ks', 'mu_max', 'synh', 'ys')
        assert model.inputs == ('b',)
    
    def test_integrate_1(self):
        #Test attributes related to integration   
        model    = dn.read_ini('_test/TestModel_1.ini')['model_1']['model']
        
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
        u      = inputs.loc[0].values
        u0     = u[0]
        assert all(u0 == [2])
        
        #Test model function
        t = 0
        f = model.func
        
        r = f(t, y, p, u0)
        assert all(r)
        
        #Test integration
        y_model, t_model = model(y, p, u, _tspan=tspan, scenario=0)
        
        assert y_model.shape == (3, 62)
    
    def test_exv_1(self):
        #Test exv function
        model  = dn.read_ini('_test/TestModel_2.ini')['model_1']['model']
        exvs   = model.exvs
        exv_1  = exvs['growth']
        assert model.states == ('x', 's', 'h')
        assert model.params == ('ks', 'mu_max', 'synh', 'ys')
        assert model.inputs == ('b',)
    
        
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
        u      = inputs.loc[0].values
        u0     = u[0]
        assert all(u0 == [2])
        
        #Test model function
        t = 0
        f = model.func
        
        r = f(t, y, p, u0)
        assert all(r)
        
        #Test integration
        y_model, t_model = model(y, p, u, _tspan=tspan, scenario=0)
        
        assert y_model.shape == (3, 62)
    
        t1 = t_model[:2]
        y1 = np.array([y, y+r]).T
        p1 = np.array([p, p]).T
        u1 = np.array([u0, u0]).T 
        
        xo1, yo1 = exv_1(t1, y1, p1, u1)
        assert all(xo1 == t_model[:2])
        assert np.isclose(yo1[0], r[0])
    
    def test_write_1(self):
        #Test writing .ini
        #Test updating
        c = '''
        [model_1]
        states = 
            x0 = [0, 1],
            x1 = [2, 3]
            
        params = 
            p0 = [0, 1],
            p1 = [2, 3]
            
        inputs =
            u0 = [[0, 0], [1, 1]],
            u1 = [[0, 1], [1, 0]]
    
        tspan =
            linspace(0, 100, 11)
        
        equations = 
            dx0 = -p0*x0 -u0*x1
            dx1 = -p1*x1 -u1*x0
            
        '''
        model_data  = dn.read_inicode(c)
        model       = model_data['model_1']['model']
        ini_section = model_data['model_1']['ini_section']
        
        ix = pd.MultiIndex.from_tuples(((0, 0), (0, 1), (1, 0), (1, 1)), names=['scenario', 'segment'])
        df = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 80]], 
                          columns = ['u0', 'u1'], 
                          index   = ix
                          )
        
        tspan = np.array([[0, 1, 2], [2, 3, 4]])
        
        #Update
        input_vals = df
        
        init_vals = df.xs(0, level=1)
        init_vals.columns = ['x0', 'x1']
        
        param_vals = df.xs(0, level=1)
        param_vals.columns = ['p0', 'p1']
        
        to_update_model_1 = {'states': init_vals,
                             'params': param_vals,
                             'inputs': input_vals
                             }
        
        new_filename   = '_test/writeModel_1.ini' 
        new_inicode    = dn.make_updated_inicode(model_data, filename=new_filename, model_1=to_update_model_1)
        new_model_data = dn.read_ini(new_filename)
        new_model      = new_model_data['model_1']['model']
        
        assert all(input_vals == new_model.input_vals)
        assert all(param_vals == new_model.param_vals)
        assert all(init_vals  == new_model.init_vals )
        
        
if __name__ == '__main__':
    T = TestModelInstance()
    T.test_read_1()
    T.test_integrate_1()
    T.test_exv_1()
    T.test_write_1()