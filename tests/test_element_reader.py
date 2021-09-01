import numpy  as np
import pandas as pd

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin                                 as dn  
import dunlin._utils_model.dun_element_reader as der
    
if __name__ == '__main__':
    remove_whitespace = lambda s: s.replace(' ', '').replace('\n', '').replace('\t', '')
    compare           = lambda s1, s2: remove_whitespace(s1) == remove_whitespace(s2)  
    
    ###############################################################################
    #Test Evaluation of Values for Substitution
    ###############################################################################
    #Test with neat string
    raw_values = 'range(0, 4, 2), 5, 6, 7'
    new_values = der.eval_sub(raw_values)
  
    assert new_values == ['0', '2', '5', '6', '7']
    
    #Test with messy string
    raw_values = 'range(0, 4,     2), 5, \n\t6, 7\n'
    new_values = der.eval_sub(raw_values)
    
    assert new_values == ['0', '2', '5', '6', '7']

    ###############################################################################
    #Test Low-Level Substitution
    ###############################################################################
    #Test horizontal
    template = 'a: [<x{i}, y{i}>, <z{ii}>]'
    h_vals   = {'i' : ['0', '1', '2'],
                'ii': ['7', '8', '9']
                }
    
    string = der.substitute_horizontal(template, h_vals)
    answer = 'a: [x0, y0, x1, y1, x2, y2, z7, z8, z9]'
    assert compare(string, answer)
    
    #Test vertical
    template = 'a{i}: [{i}, {ii}]'
    v_vals   = {'i' : ['0', '1', '2'],
                'ii': ['7', '8', '9']
                }
    
    strings = der.substitute_vertical(template, v_vals)
    answers = ['a0: [0, 7]', 'a1: [1, 8]', 'a2: [2, 9]']
    
    assert len(strings) == len(answers)
    for s, a in zip(strings, answers):
        assert compare(s, a)
        
    #Test horizontal with no h_vals
    template = 'a: [0]'
    h_vals   = {}
    
    string = der.substitute_horizontal(template, h_vals)
    answer = 'a: [0]'
    assert compare(string, answer)
    
    #Test vertical with no v_vals
    template = 'a: [0]'
    v_vals   = {}
    
    strings = der.substitute_vertical(template, v_vals)
    answers = ['a: [0]']
    assert len(strings) == len(answers)
    for s, a in zip(strings, answers):
        assert compare(s, a)
    
    # #Test horizontal with missing h_vals
    # template = 'a: [<{i}>]'
    # h_vals   = {}
    
    # try:
    #     string = der.substitute_horizontal(template, h_vals)
    # except der.DunlinShorthandError as e:
    #     assert e.num == 0
    # else:
    #     assert False
    
    # #Test vertical with missing v_vals
    # template = 'a: [{i}]'
    # v_vals   = {}
    
    # try:
    #     string = der.substitute_vertical(template, v_vals)
    # except der.DunlinShorthandError as e:
    #     assert e.num == 0
    # else:
    #     assert False
    
    # #Test horizontal with missing h_vals
    # template = 'a: [<{i}>]'
    # h_vals   = {}
    
    # try:
    #     string = der.substitute_horizontal(template, h_vals)
    # except der.DunlinShorthandError as e:
    #     assert e.num == 1
    # else:
    #     assert False
    
    # #Test vertical with missing v_vals
    # template = 'a: [{i}]'
    # v_vals   = {}
    
    # try:
    #     string = der.substitute_horizontal(template, h_vals)
    # except der.DunlinShorthandError as e:
    #     assert e.num == 1
    # else:
    #     assert False
    
    # #Test horizontal with missing h_vals
    # template = 'a: [<{i}>]'
    # h_vals   = {'z': 4}
    
    # try:
    #     string = der.substitute_horizontal(template, h_vals)
    # except der.DunlinShorthandError as e:
    #     assert e.num == 1
    # else:
    #     assert False
    
    # #Test vertical with missing v_vals
    # template = 'a: [{i}]'
    # v_vals   = {'z': 4}
    
    # try:
    #     string = der.substitute_horizontal(template, h_vals)
    # except der.DunlinShorthandError as e:
    #     assert e.num == 1
    # else:
    #     assert False
        
        
    ###############################################################################
    #Test Extraction of Template and Values
    ###############################################################################
    #Test horizontal with neat string
    element = 'a: [<x{i}, y{i}>, <z{ii}>]!i, range(0, 4, 2)!ii, 7, 8, 9'
    t, h, v = der.split_dun_element(element)
    
    t_answer = 'a: [<x{i}, y{i}>, <z{ii}>]'
    h_answer = {'i': ['0', '2'], 'ii': ['7', '8', '9']}
    v_answer = {}
    
    assert compare(t_answer, t)
    assert h_answer == h
    assert v_answer == v
    
    #Test horizontal with messy string
    element = 'a: [<x{i}, y{i}>, <z{ii}>]\n!i, range(0, \t4, 2)!\nii, 7, 8, 9'
    t, h, v = der.split_dun_element(element)
    
    assert compare(t_answer, t)
    assert h == h_answer
    assert v == v_answer
    
    #Test vertical with neat string
    element = 'a: [x{i}, y{i}, z{ii}]~i, range(0, 4, 2)~ii, 7, 8, 9'
    t, h, v = der.split_dun_element(element)
    
    t_answer = 'a: [x{i}, y{i}, z{ii}]'
    h_answer = {}
    v_answer = {'i': ['0', '2'], 'ii': ['7', '8', '9']}
    
    assert compare(t_answer, t)
    assert h_answer == h
    assert v_answer == v
    
    #Test vertical with neat string
    element = 'a: [x{i}, y{i}, z{ii}]  ~\ti, range(0\n, 4, 2)~ii, 7, 8, 9'
    t, h, v = der.split_dun_element(element)
    
    t_answer = 'a: [x{i}, y{i}, z{ii}]'
    h_answer = {}
    v_answer = {'i': ['0', '2'], 'ii': ['7', '8', '9']}
    
    assert compare(t_answer, t)
    assert h_answer == h
    assert v_answer == v
    
    #Test vertical with neat string
    element = 'a: [x{i}, y{i}, z{ii}]~i, range(0, 4, 2)~ii, 7, 8, 9'
    t, h, v = der.split_dun_element(element)
    
    t_answer = 'a: [x{i}, y{i}, z{ii}]'
    h_answer = {}
    v_answer = {'i': ['0', '2'], 'ii': ['7', '8', '9']}
    
    assert compare(t_answer, t)
    assert h_answer == h
    assert v_answer == v
    
    #Test mixed with neat string
    element = 'a: [x{i}, y{i}, z{ii}, <h{iii}>]~i, range(0, 4, 2)!iii, _j, _k~ii, 7, 8, 9'
    t, h, v = der.split_dun_element(element)
    
    t_answer = 'a: [x{i}, y{i}, z{ii}, <h{iii}>]'
    h_answer = {'iii': ['_j', '_k']}
    v_answer = {'i': ['0', '2'], 'ii': ['7', '8', '9']}
    
    assert compare(t_answer, t)
    assert h_answer == h
    assert v_answer == v
    
    #Test mixed with messy string
    element = 'a: [x{i}, y{i}, z{ii}, <h{iii}>]\n~\ni, range(0\n, 4, 2)!iii, _j,\t _k~ii\t,   7, 8, 9'
    t, h, v = der.split_dun_element(element)
    
    assert compare(t_answer, t)
    assert h_answer == h
    assert v_answer == v
    
    #Test overlapping horizontal and vertical fields
    element = 'a: [x{i}, y{i}, z{ii}, <h{i}>]~i, range(0, 4, 2)!i, _j, _k~ii, 7, 8, 9'
    t, h, v = der.split_dun_element(element)
    
    t_answer = 'a: [x{i}, y{i}, z{ii}, <h{i}>]'
    h_answer = {'i': ['_j', '_k']}
    v_answer = {'i': ['0', '2'], 'ii': ['7', '8', '9']}
    
    assert compare(t_answer, t)
    assert h_answer == h
    assert v_answer == v
    
    ###############################################################################
    #Test High-Level Substitution for dun Element
    ###############################################################################
    #Test mixed with messy string
    element = 'a: [x{i}, y{i}, z{ii}, <h{iii}>]\n~\ni, range(0\n, 4, 2)!iii, _j,\t _k~ii\t,   7, 8,'
    
    strings = der.substitute_dun(element)
    answers = ['a: [x0, y0, z7, h_j, h_k]\n', 
               'a: [x2, y2, z8, h_j, h_k]\n'
               ]
    
    assert len(strings) == len(answers)
    for s, a in zip(strings, answers):
        assert compare(s, a)
        
    #Test with missing h_vals
    element = 'a: [x{i}, y{i}, z{ii}, <h{iii}>]\n~\ni, range(0\n, 4, 2)~ii,\t   7, 8, '
    
    try:
        der.substitute_dun(element)
    except der.DunlinShorthandError as e:
        assert e.num == 0
    
    #Test with missing v_vals
    element = 'a: [x{i}, y{i}, z{ii}, <h{iii}>]\n\n!iii, _j,\t _k~ii,\t   7, 8, '
    
    try:
        der.substitute_dun(element)
    except der.DunlinShorthandError as e:
        assert e.num == 0
    
    #Test unequal length
    element = 'a: [x{i}, y{i}, z{ii}, <h{iii}>]\n~\ni, range(0\n, 4)!iii, _j,\t _k~ii\t,   7, 8, 9'
    
    try:
        der.substitute_dun(element)
    except der.DunlinShorthandError as e:
        assert e.num == 1
        
    ###############################################################################
    #Test High Level Substitution for py Elements
    ###############################################################################
    element = 'def f():\n\t@short x{i} = {i}\n\t~i, 0, 1, 2\n\t@short return [<x{i}>]!i, 0, 1, 2'
    f       = None
    result  = der.substitute_py(element)
    exec(result)
    assert f() == [0, 1, 2]
    
    ###############################################################################
    #Test Decorator and Parser
    ###############################################################################
    decorator = der.element_type
    
    @decorator('dun')
    def parse_test_dun(data):
        data['find_this'] = 'Here I am.'
        return data
    
    element = 'a: [0, 1, 2], b:[3, 4, 5]'
    result  = parse_test_dun(element)
    
    assert type(result) == dict
    assert len(result)  == 3
    assert result['a']         == [0.0, 1.0, 2.0]
    assert result['b']         == [3.0, 4.0, 5.0]
    assert result['find_this'] == 'Here I am.'
    
    @decorator('py')
    def parse_test_py(data):
        return 'def my_func():' + data #+ '\n\traise Exception("Hello World")'
    
    element = 'f:\n\t@short x{i} = {i}\n\t~i, 0, 1, 2\n\t@short return [<x{i}>]!i, 0, 1, 2'
    my_func = None
    result  = parse_test_py(element)
    exec(result['f'])
    assert my_func() == [0, 1, 2]
    
    ###############################################################################
    #Test Parser Dict
    ###############################################################################
    der.parsers['test'] = [parse_test_dun, 'test_field']
    
    func    = der.parsers['test'] 
    element = 'a: [0, 1, 2], b:[3, 4, 5]'
    result  = func[0](element)
    
    assert type(result) == dict
    assert len(result)  == 3
    assert result['a']         == [0.0, 1.0, 2.0]
    assert result['b']         == [3.0, 4.0, 5.0]
    assert result['find_this'] == 'Here I am.'
    
    
    