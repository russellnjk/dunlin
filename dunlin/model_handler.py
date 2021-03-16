import configparser as cp
import numpy        as np
import pandas       as pd
from   pathlib      import Path

###############################################################################
#Non-Standard Imports
###############################################################################
try:
    import dunlin._utils_model.model_coder as coder
    import dunlin._utils_model.utils_ini   as uti
    import dunlin._utils_model.utils_model as utm
except Exception as e:
    if Path.cwd() == Path(__file__).parent:
        import _utils_model.model_coder as coder
        import _utils_model.utils_ini   as uti
        import _utils_model.utils_model as utm
    else:
        raise e

###############################################################################
#Globals
###############################################################################
display_settings = {'verbose' : False}

###############################################################################
#.ini Constructors
###############################################################################
def read_ini(filename):
    with open(filename, 'r') as file:
        return read_inicode(file.read())
            
def read_inicode(inicode):
    '''
    Parses a string of code in .ini format. Note that Only the semicolon (';') 
    can be used to mark comments. Lines with the hash ('#') symbol will be read 
    and used as Python comments.

    Parameters
    ----------
    inicode : str
        A string of code in .ini format. Only the semicolon (';') can be used to 
        mark comments. Lines with the hash ('#') symbol will be read and used as 
        Python comments.

    Returns
    -------
    model_data : dict
        A dictionary of the data from the code.
    '''
    model_data = {}
    config     = cp.RawConfigParser(comment_prefixes=(';',))
    config.optionxform = str 
    config.read_string(inicode)
    
    for name in config.sections():
        model_data[name] = parse_model(name, config)
    return model_data

def parse_model(name, config):
    '''
    Parses data indexed under name from a configparser object.
    
    :meta private:
    '''
    data = uti.parse_section(name, config)
    
    data['model'] = Model(**data['model'])
    model         = data['model']
    
    if 'objectives' in data:
        objs                    = data['objectives']
        objs, obj_code          = coder.objectives2func(model.as_dict(), objs) 
        data['objectives']      = objs
        data['objectives_code'] = obj_code
    
    return data

###############################################################################
#Model Class
###############################################################################
class Model():
    ###############################################################################
    #Constructors
    ###############################################################################
    def __init__(self, name, states, params, inputs, equations, meta=None, tspan=None, int_args=None, modify=None, solver_args=None, use_numba=True):

        if type(name) != str:
            raise TypeError('name argument must be a str.')
        if meta is not None and type(meta) != dict:
            raise TypeError('meta argument must be a dict.')
        
        #Get names of states, params and inputs
        state_names, init_vals  = utm.read_init(states)
        param_names, param_vals = utm.read_params(params)
        input_names, input_vals = utm.read_inputs(inputs)
        tspan_                  = utm.read_tspan(tspan)
        solver_args_            = solver_args if solver_args else {'method': 'LSODA'}
        
        utm.check_names(state_names, param_names, input_names)
        
        set_attr = super().__setattr__
        
        #Attributes assigned directly
        set_attr('name',        name)
        set_attr('states',      state_names)
        set_attr('params',      param_names)
        set_attr('inputs',      input_names)
        set_attr('meta',        meta)
        set_attr('solver_args', solver_args_)
        set_attr('modify',      modify)
        set_attr('tspan',       tspan_)
        set_attr('init_vals',   init_vals)
        set_attr('param_vals',  param_vals)
        set_attr('input_vals',  input_vals),
        set_attr('equations',   equations)
        
        #Function generation
        if type(equations) == str:
            model_dict = self.as_dict()
            # use_numba=False
            func, code = coder.model2func(model_dict, use_numba=use_numba)
            set_attr('code', code)
            set_attr('func', func)
        elif callable(equations):
            set_attr('code', code)
            set_attr('func', func)
        else:
            raise TypeError('equation argument must be a function or a string that can be executed as a function.')
    
    ###############################################################################
    #Immutability
    ###############################################################################
    def __setattr__(self, attr, value):
        if attr == 'init_vals':
            states, value_ = utm.read_init(value)
            if set(states).difference(self.states):
                raise ValueError('Attempted to assign init_vals but names of states do not match those of model.')
            super().__setattr__(attr, value_[list(self.states)])
        elif attr == 'param_vals':
            names, value_ = utm.read_params(value)
            if set(names).difference(self.params):
                raise ValueError('Attempted to assign param_vals but names of params do not match those of model.')
            super().__setattr__(attr, value_[list(self.params)])
        elif attr == 'input_vals':
            names, value_ = utm.read_inputs(value)
            if set(names).difference(self.params):
                raise ValueError('Attempted to assign input_vals but names of inputs do not match those of model.')
            super().__setattr__(attr, value_[list(self.inputs)])
        elif attr == 'tspan':
            value_ = utm.read_tspan(value)
            super().__setattr__(attr, value_)
        elif attr == 'modify':
            if value is None:
                pass
            elif not callable(value):
                msg = 'The "modify" attribute for Model "{}" must be a function.'
                raise TypeError(msg.format(self.name))
            elif value.__code__.co_argcount < 4 and not self.inputs:
                msg = 'The "modify" attribute for Model "{}" must have at least 4 arguments: init, params, scenario, segment.'
                raise ValueError(msg.format(self.name))
            elif value.__code__.co_argcount < 5 and self.inputs:
                msg = 'The "modify" attribute for Model "{}" must have at least 5 arguments: init, params, inputs, scenario, segment.'
                raise ValueError(msg.format(self.name))
            super().__setattr__(attr, value)
        elif attr in ['meta', 'solver_args']:
            super().__setattr__(attr, value)
        else:
            msg = "Model object's {} attribute is fixed.".format(attr)
            raise AttributeError(msg)
            
    ###############################################################################
    #Export
    ###############################################################################
    def as_dict(self):
        model_dict = {'name'      : self.name,
                      'states'    : self.states,
                      'params'    : self.params,
                      'inputs'    : self.inputs,
                      'meta'      : self.meta,
                      'equations' : self.equations
                      }
        return model_dict
    
    def export_code(self, filename=''):
        filename_ = filename if filename else 'model_{}.py'.format(self.name)
        with open(filename_, 'w') as file:
            file.write(self.code)
    
    ###############################################################################
    #Printing
    ###############################################################################
    def __str__(self):
        global display_settings
        if display_settings['verbose']:
            d = self.as_dict()
            s = 'Model({})'.format(d)
            return s
        else:
            s = 'Model({})'.format(self.name)
            return s

    def __repr__(self):
        return self.__str__()

if __name__ == '__main__':    
    import _utils_model.integration as itg
    # #Read .ini
    # #Case 1: Basic arguments
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
    
    
    
    
    