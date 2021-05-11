import configparser as cp
import numpy        as np
import pandas       as pd
from   pathlib      import Path

###############################################################################
#Non-Standard Imports
###############################################################################
try:
    import dunlin._utils_model.model_coder  as coder
    import dunlin._utils_model.ini_reader   as uir
    import dunlin._utils_model.ini_writer   as uiw 
    import dunlin._utils_model.attr_checker as uac
    import dunlin._utils_model.integration  as itg
except Exception as e:
    if Path.cwd() == Path(__file__).parent:
        import _utils_model.model_coder  as coder
        import _utils_model.ini_reader   as uir
        import _utils_model.ini_writer   as uiw 
        import _utils_model.attr_checker as uac
        import _utils_model.integration as itg
    else:
        raise e

###############################################################################
#Globals
###############################################################################
display_settings   = {'verbose' : False}
update_ini_section = uiw.update_ini_section
make_config        = uiw.make_config

###############################################################################
#.ini Constructors
###############################################################################
def read_ini(filename, **kwargs):
    '''
    Reads a .ini file and passes the string and keyword arguments to 
    dunlin.model_handler.read_inicode.

    Parameters
    ----------
    filename : str
        Name of the file to be read.
    **kwargs : dict
        Arguments for dunlin.model_handler.read_inicode.

    Returns
    -------
    model_data : dict
        A dictionary of model data after parsing.
    config : configparser.ConfigParser
        The ConfigParser object generated in reading the string/file. This is 
        returned only if the return_config argument is set to True.
    '''
    with open(filename, 'r') as file:
        return read_inicode(file.read(), **kwargs)
            
def read_inicode(inicode, append_model_name=False, return_config=False, **kwargs):
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
    append_model_name : bool, optional
        Appends the name of a model to each of its parameters. The default is 
        False.
    return_config : bool, optional
        Returns the ConfigParser object generated in reading the string/file if 
        True.
    **kwargs : dict
        Keyword arguments for instantiating the ConfigParser.

    Returns
    -------
    model_data : dict
        A dictionary of model data after parsing.
    config : configparser.ConfigParser
        The ConfigParser object generated in reading the string/file. This is 
        returned only if the return_config argument is set to True.
    '''
    ini_args   = {'comment_prefixes':(';',), 'interpolation':cp.ExtendedInterpolation()}
    ini_args   = {**ini_args, **kwargs}
    config     = cp.ConfigParser(**ini_args)
    
    config.optionxform = str 
    config.read_string(inicode)
    
    model_data = read_config(config, append_model_name)
    
    if return_config:
        return model_data, config
    else:
        return model_data

def read_config(config, append_model_name=False):
    '''
    Reads a configparser.ConfigParser object and parses the data.    

    Parameters
    ----------
    config : configparser.ConfigParser
        The object to be parsed.
    append_model_name : bool, optional
        Appends the name of a model to each of its parameters. The default is 
        False.

    Returns
    -------
    model_data : dict
        A dictionary of model data after parsing.
    '''
    model_data = {}
    for name in config.sections():
        if name[0] == '_' or name == config.default_section:
            continue
        model_data[name] = parse_model(name, config, append_model_name)
    return model_data

def parse_model(name, config, append_model_name=False):
    '''
    Parses data indexed under name from a configparser exvect.
    
    :meta private:
    '''
    data          = uir.parse_section(name, config)
    data['model'] = Model(append_model_name=append_model_name, **data['model'])
    
    return data

###############################################################################
#Model Class
###############################################################################
class Model():
    ###############################################################################
    #Constructors
    ###############################################################################    
    def __init__(self,             name,             states, 
                 params,           inputs,           equations, 
                 tspan=None,       exv_eqns=None,    modify_eqns=None, 
                 solver_args=None, meta=None,
                 append_model_name=False
                 ):

        #Checks
        if type(name) != str:
            raise TypeError('name argument must be a str.')
        if meta is not None and type(meta) != dict:
            raise TypeError('meta argument must be a dict.')
        if type(equations) != str:
            raise TypeError('equations must be a string.')
        
        #Get names of states, params and inputs with checks
        state_names, init_vals  = uac.read_init(states)
        param_names, param_vals = uac.read_params(params)
        input_names, input_vals = uac.read_inputs(inputs)
        tspan_                  = uac.read_tspan(tspan)
        solver_args_            = solver_args if solver_args else {'method': 'LSODA'}
        
        uac.check_names(state_names, param_names, input_names)
        
        #Append model name if required
        param_names_ = [p + '_' + name] if append_model_name else param_names
        
        set_attr = super().__setattr__
        
        #Descriptional attributes
        set_attr('name',        name)
        set_attr('states',      state_names)
        set_attr('params',      param_names_)
        set_attr('inputs',      input_names)
        set_attr('meta',        meta)
        set_attr('solver_args', solver_args_)
        set_attr('tspan',       tspan_)
        set_attr('init_vals',   init_vals)
        set_attr('param_vals',  param_vals)
        set_attr('input_vals',  input_vals)
        set_attr('equations',   equations)
        set_attr('code',        None)
        set_attr('func',        None)
        set_attr('exv_eqns',    exv_eqns)
        set_attr('exv_code' ,   None)
        set_attr('exvs',        {})
        set_attr('modify_eqns', modify_eqns)
        set_attr('modify_code', None) 
        set_attr('modify',      None)
        set_attr('appended',    append_model_name)
        
        
        #Functions and code
        model_dict = {'name'      : name,
                      'states'    : state_names,
                      'params'    : param_names,
                      'inputs'    : input_names,
                      'equations' : equations
                      }
        if equations:
            func, code = coder.model2func(model_dict)
            set_attr('code', code)
            set_attr('func', func)
            
        if exv_eqns:
            exvs, exv_code = coder.exvs2func(model_dict, exv_eqns)
            set_attr('exvs',     exvs)
            set_attr('exv_code', exv_code)
        
        if modify_eqns:
            mod_func, mod_code = coder.modify2func(model_dict, modify_eqns, func)
            set_attr('modify',      mod_func)
            set_attr('modify_code', mod_code)
        
    ###############################################################################
    #Immutability
    ###############################################################################
    def __setattr__(self, attr, value):
        arg_msg     = 'The "{}" attribute for Model {} must have {} arguments: {}.'
        func_msg    = 'The "{}" attribute for Model "{}" must be a function.'
        dict_msg    = 'The "{}" attribute for Model {} must be a dict.'
        if attr == 'init_vals':
            states, value_ = uac.read_init(value)
            if set(states).difference(self.states):
                raise ValueError('Attempted to assign init_vals but names of states do not match those of model.')
            super().__setattr__(attr, value_[list(self.states)])
        elif attr == 'param_vals':
            names, value_ = uac.read_params(value)
            if set(names).difference(self.params):
                raise ValueError('Attempted to assign param_vals but names of params do not match those of model.')
            super().__setattr__(attr, value_[list(self.params)])
        elif attr == 'input_vals':
            names, value_ = uac.read_inputs(value)
            if set(names).difference(self.params):
                raise ValueError('Attempted to assign input_vals but names of inputs do not match those of model.')
            super().__setattr__(attr, value_[list(self.inputs)])
        elif attr == 'tspan':
            value_ = uac.read_tspan(value)
            super().__setattr__(attr, value_)
        elif attr == 'exvs':
            if type(value) != dict and value is not None:
                raise TypeError(dict_msg(attr, self.name))
            super().__setattr__(attr, value)
        elif attr == 'modify':
            if value is None:
                pass
            elif not callable(value):
                raise TypeError(func_msg.format(attr, self.name))
            elif value.__code__.co_argcount < 5 and not self.inputs:
                raise ValueError(arg_msg.format(attr, self.name, 5, 'model_func, init, params, scenario, segment'))
            elif value.__code__.co_argcount < 6 and self.inputs:
                print(attr, self.name, 6, 'model_func')
                raise ValueError(arg_msg.format(attr, self.name, 6, 'model_func, init, params, inputs, scenario, segment'))
            super().__setattr__(attr, value)
        elif attr in ['meta', 'solver_args']:
            super().__setattr__(attr, value)
        elif hasattr(self, attr):
            msg = "Model object's {} attribute is fixed.".format(attr)
            raise AttributeError(msg)
        else:
            msg = "Model object has no attribute called: {}".format(attr)
            raise AttributeError(msg)
    
    ###############################################################################
    #Export
    ###############################################################################
    def export_func(self, filename=''):
        filename_ = filename if filename else 'model_{}.py'.format(self.name)
        with open(filename_, 'w') as file:
            file.write(self.code)
    
    def export_exv(self, filename=''):
        filename_ = filename if filename else 'exvs_{}.py'.format(self.name)
        with open(filename_, 'w') as file:
            file.write(self.exv_code)
    
    def export_modify(self, filename):
        filename_ = filename if filename else 'modify_{}.py'.format(self.name)
        with open(filename_, 'w') as file:
            file.write(self.modify_code)
    
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
    
    ###############################################################################
    #Integration
    ###############################################################################
    def __call__(self, init, params, inputs, scenario=None, args=(), _tspan=None, **_solver_args):
        tspan       = self.tspan       if _tspan       is None else _tspan
        solver_args = self.solver_args if _solver_args is None else _solver_args
        
        y_model, t_model = itg.piecewise_integrate(self.func, 
                                                   tspan, 
                                                   init, 
                                                   params, 
                                                   inputs, 
                                                   scenario = scenario, 
                                                   modify   = self.modify,
                                                   args     = args,
                                                   **solver_args
                                                   )
        
        return y_model, t_model
    
    # def integrate_array(self, scenario, estimate, _tspan=None, _init=None, _params=None, _inputs=None, _solver_args=None):
    #     tspan       = self.tspan                    if _tspan       is None else _tspan
    #     y           = self.init_vals.loc[scenario]  if _init        is None else _init
    #     p           = self.param_vals.loc[estimate] if _params      is None else _params
    #     u           = self.input_vals.loc[scenario] if _inputs      is None else _inputs
    #     solver_args = self.solver_args              if _solver_args is None else _solver_args
        
    #     y_model, t_model = itg.piecewise_integrate(self.func, tspan, y, p, u, scenario=scenario, **solver_args)
        
    #     return y_model, t_model
    
if __name__ == '__main__':    
    import _utils_model.integration as itg
    # #Read .ini
    # #Case 1: Basic arguments
    model = read_ini('_test/TestModel_1.ini')['model_1']['model']
    assert model.states == ('x', 's', 'h')
    assert model.params == ('ks', 'mu_max', 'synh', 'ys')
    assert model.inputs == ('b',) 
    
    #Case 2: With exv
    model = read_ini('_test/TestModel_2.ini')['model_1']['model']
    assert model.states == ('x', 's', 'h')
    assert model.params == ('ks', 'mu_max', 'synh', 'ys')
    assert model.inputs == ('b',)
    
    exvs = model.exvs#read_ini('_test/TestModel_2.ini')['model_1']['exvs']
    assert len(exvs) == 1
    
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
    u      = inputs.loc[0].values
    u0     = u[0]
    assert all(u0 == [2])
    
    #Test model function
    t = 0
    f = model.func
    
    r = f(t, y, p, u0)
    assert all(r)
    
    #Test integration
    y_model, t_model = itg.piecewise_integrate(model.func, tspan, y, p, u, scenario=0)
    assert y_model.shape == (3, 62)
    
    #Test exv function
    model  = read_ini('_test/TestModel_2.ini')['model_1']['model']
    exvs   = model.exvs
    exv_1  = exvs['growth']
    
    t1 = t_model[:2]
    y1 = np.array([y, y+r]).T
    p1 = np.array([p, p]).T
    u1 = np.array([u0, u0]).T 
    
    xo1, yo1 = exv_1(t1, y1, p1, u1)
    assert all(xo1 == t_model[:2])
    assert np.isclose(yo1[0], r[0])
    
    
    
    
    