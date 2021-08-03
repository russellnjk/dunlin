import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin                              as dn 
import dunlin._utils_model.dun_file_reader as dfr
import dunlin._utils_model.ode_coder       as odc

if __name__ == '__main__':
    import dun_file_reader as dfr

    dun_data0 = dfr.read_file('dun_test_files/M20.dun')
    dun_data1 = dfr.read_file('dun_test_files/M21.dun')
    model_data0 = dun_data0['M1']
    model_data1 = dun_data1['M2']
    model_data2 = dun_data1['M3']
    
    ###############################################################################
    #Part 1: Low Level Code Generation
    ###############################################################################
    funcs  = model_data0['funcs']
    vrbs   = model_data0['vrbs']
    rxns   = model_data0['rxns']
    states = model_data0['states']
    
    #Test func def
    name, args = 'test_func', ['a', 'b']
    
    code      = odc.make_def(name, *args)
    test_func = f'{code}\n\treturn [a, b]'
    exec(test_func)
    a, b = 1, 2
    assert test_func(a, b) == [1, 2]
    
    #Test code generation for local functions
    code      = odc.funcs2code(funcs)
    test_func = f'def test_func(v, x, k):\n{code}\n\treturn MM(v, x, k)'
    exec(test_func)
    assert test_func(2, 4, 6) == 0.8
    
    #Test local variable
    code      = odc.vrbs2code(vrbs)
    test_func = f'def test_func(x2, k1):\n{code}\n\treturn sat2'
    exec(test_func)
    assert test_func(1, 1) == 0.5
    
    #Parse single reaction
    stripper = lambda *s: ''.join(s).replace(' ', '').strip()
    r = odc._parse_rxn(*rxns['r0'])
    
    assert {'x0': '-1', 'x1': '-2', 'x2': '+1'} == r[0]
    assert stripper(rxns['r0'][1], '-', rxns['r0'][2]) == stripper(r[1])
    
    r = odc._parse_rxn(*rxns['r1'])
    
    assert {'x2': '-1', 'x3': '+1'} == r[0]
    assert stripper(rxns['r1'][1]) == stripper(r[1])
    
    r = odc._parse_rxn(*rxns['r2'])
    
    assert {'x3': '-1'} == r[0]
    assert stripper(rxns['r2'][1]) == stripper(r[1])
    
    #Test code generation for multiple reactions
    code = odc.rxns2code(model_data0)
    
    MM        = lambda v, x, k: 0
    sat2      = 0.5
    test_func = f'def test_func(x0, x1, x2, x3, x4, p0, p1, p2, p3, p4):\n{code}\treturn [d_x0, d_x1, d_x2, d_x3, d_x4]'
    exec(test_func)
    r = test_func(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    assert r == [-1.0, -2.0, 0.5, -0.5, 1]
    
    #Test code generation for hierarchical models
    #We need to create the "submodel"
    MM         = lambda v, x, k: 0
    code       = odc.rxns2code(model_data1)
    test_func0 = 'def model_M2(*args): return np.array([1, 1])'
    exec(test_func0)
    
    code = odc.rxns2code(model_data2)
    test_func = f'def test_func(t, x0, x1, x2, x3, p0, p1, p2, p3, k2):\n{code}\treturn [d_x0, d_x1, d_x2, d_x3]'
    exec(test_func)
    r = test_func(0, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    assert r == [-1, 2, 1, 1]
    
    temp                          = dun_data1['M3']['rxns']['r1']
    dun_data1['M3']['rxns']['r1'] = {'submodel': 'M2', 
                                     'substates': {'xx0': 'x1', 'xx1': 'x2'}, 
                                     'subparams': {'pp0' : 'p0', 'pp1' : 'p1', 'kk1': 'k2'}
                                     }
    
    try:
        code = odc.rxns2code(model_data2)
    except NotImplementedError as e:
        assert True
    else:
        assert False
    dun_data1['M3']['rxns']['r1'] = temp
    
    ###############################################################################
    #Part 2: High Level Code Generation
    ###############################################################################
    template0 = odc.make_template(model_data0)
    template1 = odc.make_template(model_data1)
    template2 = odc.make_template(model_data2)
    
    params  = model_data0['params']
    exvs    = model_data0['exvs']
    events  = model_data0['events'] 
    modify  = model_data0['modify'] 
    
    #Generate code for ode rhs
    code      = odc.rhs2code(template0, model_data0)[1]
    test_func = code.replace('model_M1', 'test_func')

    exec(test_func)
    t  = 0 
    y  = np.ones(5)
    p  = pd.DataFrame(params).values[0]
    dy = test_func(t, y, p)
    assert all( dy == np.array([-0.5, -1,  0,  -1.5 , 2]) )
    
    #Generate code for exv    
    codes     = odc.exvs2code(template0, model_data0)
    test_func = codes['r0'][1].replace('exv_M1_r0', 'test_func')

    exec(test_func)
    t  = np.array([0, 1])
    y  = np.ones((5, 2))
    p  = pd.DataFrame(params).values[0]
    r  = test_func(t, y, p)
    assert all(r == 0.5)
    
    #Generate code for single event trigger
    trigger = events['e0'][0] 
    
    code      = odc.trigger2code('e0', trigger, template0, model_data0)[1]
    test_func = code.replace('trigger_M1_e0', 'test_func')
    exec(test_func)
    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = test_func(t, y, p)
    assert r == 0.5
    
    #Generate code for single event assignment
    assignment = events['e0'][1] 
    
    code      = odc.assignment2code('e0', assignment, template0, model_data0)[1]
    test_func = code.replace('assignment_M1_e0', 'test_func')
    exec(test_func)
    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = test_func(t, y, p)
    assert r[0][0]              == 5
    assert r[1][0]              == 0.5
    
    #Generate code for single event
    codes = odc.event2code('e0', template0, model_data0)
    
    test_func = codes['trigger'][1].replace('trigger_M1_e0', 'test_func')
    exec(test_func)
    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = test_func(t, y, p)
    assert r == 0.5
    
    test_func = codes['assignment'][1].replace('assignment_M1_e0', 'test_func')
    exec(test_func)
    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = test_func(t, y, p)
    assert r[0][0]              == 5
    assert r[1][0]              == 0.5
    
    #Generate code for all events
    codes = odc.events2code(template0, model_data0)
    
    test_func = codes['e0']['trigger'][1].replace('trigger_M1_e0', 'test_func')
    exec(test_func)
    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = test_func(t, y, p)
    assert r == 0.5
    
    test_func = codes['e0']['assignment'][1].replace('assignment_M1_e0', 'test_func')
    exec(test_func)
    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = test_func(t, y, p)
    assert r[0][0]              == 5
    assert r[1][0]              == 0.5
    
    #Generate modify 
    code      = odc.modify2code(template0, model_data0)[1]
    test_func = code.replace('modify_M1', 'test_func')
    exec(test_func)
    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = test_func(y, p, scenario=1)
    assert all( r[0] == np.array([10, 1, 1, 1, 1]) )
    assert all( r[1] == p)
    
    ###############################################################################
    #Part 3A: Function Generation
    ###############################################################################
    #Generate single function from code
    code      = 'x = lambda t: t+1'
    scope     = {}
    test_func = odc.code2func(['x', code])
    assert test_func(5) == 6
    
    #Generate multiple functions from codes
    #The second function requires access to the first one
    codes     = {'fx': ['x', 'def x(t):\n\treturn t+1'], 
                  'fy': ['y', 'def y(t):\n\treturn x(t)+2']
                  }
    r         = odc.code2func(codes)
    test_func = r['fx']
    assert test_func(5) == 6
    test_func = r['fy']
    assert test_func(5) == 8
    
    ###############################################################################
    #Part 3B: Function Generation
    ###############################################################################
    template0 = odc.make_template(model_data0)
    template1 = odc.make_template(model_data1)
    template2 = odc.make_template(model_data2)
    
    params  = model_data0['params']
    exvs    = model_data0['exvs']
    events  = model_data0['events'] 
    modify  = model_data0['modify'] 
    
    #Generate rhs function
    code, func = odc.rhs2func(template0, model_data0)
    t  = 0 
    y  = np.ones(5)
    p  = pd.DataFrame(params).values[0]
    dy = func(t, y, p)
    assert all( dy == np.array([-0.5, -1,  0,  -1.5 , 2]) )
    
    #Generate exv functions
    codes, funcs = odc.exvs2func(template0, model_data0)
    code, func   = codes['r0'], funcs['r0']
    
    t  = np.array([0, 1])
    y  = np.ones((5, 2))
    p  = pd.DataFrame(params).values[0]
    r  = func(t, y, p)
    assert all(r == 0.5)
    
    #Generate event functions for one event
    codes, funcs = odc.event2func('e0', template0, model_data0)
    
    func = funcs['trigger']

    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = func(t, y, p)
    assert r == 0.5
    
    func = funcs['assignment']

    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = func(t, y, p)
    assert r[0][0]              == 5
    assert r[1][0]              == 0.5
    
    #Generate event functions for all events
    codes, funcs = odc.events2func(template0, model_data0)
    
    func = funcs['e0']['trigger']

    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = func(t, y, p)
    assert r == 0.5
    
    func = funcs['e0']['assignment']

    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = func(t, y, p)
    assert r[0][0]              == 5
    assert r[1][0]              == 0.5
    
    #Generate modify 
    code, func = odc.modify2func(template0, model_data0)
    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = func(y, p, 1)
    assert all( r[0] == np.array([10, 1, 1, 1, 1]) )
    assert all( r[1] == p)
    
    ###############################################################################
    #Part 4: Top Level Functions
    ###############################################################################
    #Create functions from dun_data
    func_data = odc.make_ode_data(model_data0)
    
    #Generate rhs function
    code, func = func_data['rhs']
    t  = 0 
    y  = np.ones(5)
    p  = pd.DataFrame(params).values[0]
    dy = func(t, y, p)
    assert all( dy == np.array([-0.5, -1,  0,  -1.5 , 2]) )
    
    #Generate exv functions
    codes, funcs = func_data['exvs']
    code, func   = codes['r0'], funcs['r0']
    
    t  = np.array([0, 1])
    y  = np.ones((5, 2))
    p  = pd.DataFrame(params).values[0]
    r  = func(t, y, p)
    assert all(r == 0.5)
    
    #Generate event functions for all events
    codes, funcs = func_data['events']
    
    func = funcs['e0']['trigger']

    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = func(t, y, p)
    assert r == 0.5
    
    func = funcs['e0']['assignment']

    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = func(t, y, p)
    assert r[0][0]              == 5
    assert r[1][0]              == 0.5
    
    #Generate modify 
    code, func = func_data['modify']
    t  = 10
    y  = np.array([0, 1, 1, 1, 1])
    p  = pd.DataFrame(params).values[0]
    r  = func(y, p, 1)
    assert all( r[0] == np.array([10, 1, 1, 1, 1]) )
    assert all( r[1] == p)
    