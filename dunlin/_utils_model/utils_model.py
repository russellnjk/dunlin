import numpy   as np
import pandas  as pd
from   pathlib import Path

###############################################################################
#Non-Standard Imports
###############################################################################
try:
    import dunlin._utils_model.model_coder as coder
except Exception as e:
    if Path.cwd() == Path(__file__).parent:
        import model_coder as coder
    elif Path.cwd() == Path(__file__).parent.parent:
        import _utils_model.model_coder as coder
    else:
        raise e
        
###############################################################################
#Error Message Formatter
###############################################################################
def wrap_try(name=''):
    msg = 'Could not read {} data for instantiating Model.'.format(name)
    def wrapper(func):
        def helper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                args   = (msg,) + e.args
                args   = '\n'.join(args)
                e.args = (args, )
                raise e
        return helper
    return wrapper

###############################################################################
#Input Parsers
###############################################################################
# @wrap_try('equations')
# def read_equations(equations, solver='odeint'):
#     #Function generation
#     if callable(equations):
#         code = ''
#         func = equations
        
#     elif type(equations) == str:
#         model_dict = self.as_dict()
#         func, code = coder.model2func(model_dict, use_numba=False)
        
        
      
#     else:
#         raise TypeError('equation argument must be a function or a string that can be executed as a function.')
    
#     return code, func
    
@wrap_try('init')        
def read_init(init_data):
    if type(init_data) in [list, tuple]:
        states    = tuple(init_data)
        init_vals = None
        
    elif type(init_data) == dict:
        states    = tuple(init_data.keys())
        init_vals = {key: np.array(value) for key, value in init_data.items()}
        init_vals = pd.DataFrame.from_dict(init_vals).astype(np.float64)
        
    elif type(init_data) == pd.DataFrame:
        states    = tuple(init_data.columns)
        init_vals = init_data.astype(np.float64)
    
    elif type(init_data) == pd.Series:
        states    = tuple(init_data.index)
        init_vals = pd.DataFrame(init_data).T.astype(np.float64)
    
    else:
        raise TypeError('Argument must be a dict, pandas.DataFrame or pandas.Series.')
    
    if type(init_vals) == pd.DataFrame:
        init_vals.index.name = 'scenario'
        init_vals            = init_vals[list(states)]
        
    return states, init_vals

@wrap_try('param')  
def read_params(param_data):
    if type(param_data) in [list, tuple]:
        names      = tuple(param_data)
        param_vals = None
        
    elif type(param_data) == dict:
        names      = tuple(param_data.keys())
        param_vals = {key: np.array(value) for key, value in param_data.items()}
        param_vals = pd.DataFrame.from_dict(param_vals).astype(np.float64)
        
    elif type(param_data) == pd.DataFrame:
        names      = tuple(param_data.columns)
        param_vals = param_data.astype(np.float64)
    
    elif type(param_data) == pd.Series:
        names      = tuple(param_data.index)
        param_vals = pd.DataFrame(param_data, dtype=np.float64).T
    
    else:
        raise TypeError('Argument must be a dict, pandas.DataFrame or pandas.Series.')
    
    if type(param_vals) == pd.DataFrame:
        param_vals.index.name = 'estimate'
        param_vals            = param_vals[list(names)]
        
    return names, param_vals

@wrap_try('model input')  
def read_inputs(input_data):
    if input_data is None:
        names      = ()
        input_vals = None
        
    elif type(input_data) in [list, tuple]:
        names      = tuple(input_data)
        input_vals = None
        
    elif type(input_data) == dict:
        names  = [key[0] for key in input_data.keys()]
        names  = tuple(dict.fromkeys(names))
        input_vals = pd.DataFrame.from_dict(input_data).astype(np.float64)
        input_vals = input_vals.stack(level=1)
    
    elif type(input_data) == pd.DataFrame:
        names  = input_data.columns.get_level_values(0)
        names  = tuple(dict.fromkeys(names))
        input_vals = input_data.astype(np.float64)
        
    elif type(input_data) == pd.Series:
        names  = tuple(input_data.index.get_level_values(0))
        names  = tuple(dict.fromkeys(names))
        input_vals = pd.DataFrame(input_data, dtype=np.float64).T
        input_vals = input_vals.stack(level=1)
    
    else:
        raise TypeError('Argument must be a dict, pandas.DataFrame or pandas.Series.')    
    
    if type(input_vals) == pd.DataFrame:
        input_vals.index.names = ['scenario', 'segment']
        input_vals             = input_vals[list(names)]

    return names, input_vals

@wrap_try('tspan')  
def read_tspan(tspan):
    return [np.array(segment) for segment in tspan]

###############################################################################
#Namespace Checks
###############################################################################
def check_names(states, params, inputs):
    msg0 = 'Detected a potential clash in namespace: {}.\n'
    msg1 = '{} has been reserved as a derivative.'
    
    def helper(name, variables):
        repeated = set()
        for x in variables:
            if type(x) != str:
                raise TypeError('{} can onlt contain str. {} is a {}'.format(name, x, type(x)))
            
            elif '__' in x:
                raise ValueError('Variable names cannot contain __')
            
            elif x == 't':
                raise ValueError('t has been reserved for time.')
                
            elif variables.count(x) > 1:
                repeated.update(x)
        
        if repeated:
            raise ValueError('{} appeared more than once in {}'.format(repeated, name))
        
    all_vars = states + params + inputs if inputs else states + params
    
    helper('states', states)
    helper('params', params)
    helper('inputs', inputs)
    helper('model variables', all_vars)
       
    for name in states:
        diff = 'd' + name 
        if diff in all_vars:
            raise ValueError(msg0.format(diff) + msg1.format(diff, name))
    
    return 

if __name__ == '__main__':
    #Test namespace checks
    states, params, inputs = ('a', 'b'), ('c', 'd'), ('e',) 
    try:
        check_names(states, params, inputs)
        assert True
    except:
        assert False
    
    states, params, inputs = ('a', 2), ('c', 'd'), ('e',) 
    try:
        check_names(states, params, inputs)
        assert False
    except TypeError:
        assert True
    except:
        assert False
    
    states, params, inputs = ('a', 'da'), ('c', 'd'), ('e',) 
    try:
        check_names(states, params, inputs)
        assert False
    except ValueError:
        assert True
    except:
        assert False
    
    states, params, inputs = ('a', 'b'), ('c', 'a'), ('e',) 
    try:
        check_names(states, params, inputs)
        assert False
    except ValueError:
        assert True
    except:
        assert False
        
    #Test reading init
    init1 = {'a': [1, 2], 'b': [3, 4]}
    init2 = pd.DataFrame.from_dict(init1)
    init3 = init2.iloc[0]    
    
    #Case 1: dict
    r, r_vals = read_init(init1)
    assert r == ('a', 'b')
    assert all(r_vals['a'] == [1, 2])
    assert all(r_vals['b'] == [3, 4])

    #Case 2: DataFrame
    r, r_vals = read_init(init2)
    assert r == ('a', 'b')
    assert all(r_vals['a'] == [1, 2])
    assert all(r_vals['b'] == [3, 4])
    
    #Case 3: Series
    r, r_vals = read_init(init3)
    assert r == ('a', 'b')
    assert all(r_vals['a'] == [1])
    
    #Test reading params
    params1 = {'a': [1, 2], 'b': [3, 4]}
    params2 = pd.DataFrame.from_dict(params1)
    params3 = params2.iloc[0] 
    
    #Case 1: dict
    r, r_vals = read_params(params1)
    assert r == ('a', 'b')
    assert all(r_vals['a'] == [1, 2])
    assert all(r_vals['b'] == [3, 4])

    #Case 2: DataFrame
    r, r_vals = read_params(params2)
    assert r == ('a', 'b')
    assert all(r_vals['a'] == [1, 2])
    assert all(r_vals['b'] == [3, 4])
    
    #Case 3: Series
    r, r_vals = read_params(params3)
    assert r == ('a', 'b')
    assert all(r_vals['a'] == [1])
    
    #Test reading inputs
    inputs1 = {('a', 0): [1, 2], 
               ('b', 0): [3, 4],
               ('a', 1): [5, 6],
               ('b', 1): [7, 8],
               }
    inputs2 = pd.DataFrame(inputs1).stack(level=1)
    inputs3 = pd.DataFrame(inputs1).iloc[0]
    
    #Case 1: dict
    r, r_vals = read_inputs(inputs1)
    assert r == ('a', 'b')
    assert all(r_vals['a'][0] == [1, 5])
    assert all(r_vals['a'][1] == [2, 6])
    assert all(r_vals['b'][0] == [3, 7])
    assert all(r_vals['b'][1] == [4, 8])
    
    #Case 2: DataFrame
    r, r_vals = read_inputs(inputs2)
    assert r == ('a', 'b')
    assert all(r_vals['a'][0] == [1, 5])
    assert all(r_vals['a'][1] == [2, 6])
    assert all(r_vals['b'][0] == [3, 7])
    assert all(r_vals['b'][1] == [4, 8])
    
    #Case 3: Series
    r, r_vals = read_inputs(inputs3)
    assert r == ('a', 'b')
    assert all(r_vals['a'][0] == [1, 5])
    assert all(r_vals['b'][0] == [3, 7])
    