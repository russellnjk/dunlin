import addpath
import dunlin as dn
import dunlin._utils_model.dun_string_reader as dsr

if __name__ == '__main__':
    
    ###############################################################################
    #DN Subsection Code Strings without Errors
    ###############################################################################
    def test_string_without_error(test_string, answer, **kwargs):
        r = dsr.read_dun(test_string, **kwargs)
        assert r == answer
        
    #Test dict values
    test_strings = [['x : 0', 
                      {'x': 0}],
                    ['x : 0, y : 1', 
                      {'x': 0, 'y': 1}],
                    ['x : 0, y : 1, z : [a : 2, b : 3]', 
                      {'x': 0, 'y': 1, 'z': {'a': 2, 'b': 3}}],
                    ['x : 0, y : 1, z : [a : [c : 2, d : 3 ], b : [e : 4, f : 5]]',
                      {'x': 0, 'y': 1, 'z': {'a': {'c': 2, 'd': 3}, 'b': {'e': 4, 'f': 5}}}],
                    ['x : 0, y : 1, z : [a : [c : 2, d : 3 ], b : [e : 4, f : 5]], j : 6',
                      {'x': 0, 'y': 1, 'z': {'a': {'c': 2, 'd': 3}, 'b': {'e': 4, 'f': 5}}, 'j': 6}]
                    ]
    
    for test_string, answer in test_strings:
        test_string_without_error(test_string, answer)
        
    #Test list values and mixing
    test_strings = [['x : [0, 1]',
                      {'x': [0, 1]}],
                    ['x : [0, 1]*2, y : [0, 1, 2]',
                      {'x': [0, 1], 'y': [0, 1, 2]}],
                    ['x : [0, 1], y : [[0, 1, 2], [0, 1, 2]]',
                      {'x': [0, 1], 'y': [[0, 1, 2], [0, 1, 2]]}],
                    ['x : [0, 1], y : [[0, 1, 2], [0, 1, 2]], z : [a : 3, b : 4]',
                      {'x': [0, 1], 'y': [[0, 1, 2], [0, 1, 2]], 'z': {'a': 3, 'b': 4}}],
                    ]
    
    for test_string, answer in test_strings:
        test_string_without_error(test_string, answer)
        
    ###############################################################################
    #DN Subsection Code Strings with Errors
    ###############################################################################
    def read_dun_with_error(test_string, num, **kwargs):
        expected : False
        try:
            dsr.read_dun(test_string, **kwargs)
        except dsr.DunlinStringError as e:
            expected = e.num == num
        except Exception as e:
            raise e
        assert expected
            
    #Test inconsistent data types
    test_strings = ['x : [0, g : 1]',
                    'x : [g : 0, 1], y : [0, 1, 2]',
                    'x : [g : 0, h : 1], y : [0, j : 1, 2]',
                    ]
    
    for test_string in test_strings:
        read_dun_with_error(test_string, 1)
        
    #Test for values outside brackets
    test_strings = ['x : 0 [a : 1, b : 2]',
                    'x : [a : 1, b : 2] 0',
                    'x : [a : 1, b : 2], y : 0 [1, 2]'
                    ]
    
    for test_string in test_strings:
        read_dun_with_error(test_string, 2)
        
    #Test for top level list
    test_strings = ['[0, 1, 2]',
                    '0, 1, 2'
                    ]
    
    for test_string in test_strings:
        read_dun_with_error(test_string, 3)
        
    #Test for missing brackets
    test_strings = ['x : [0,',
                    'x : [0, 1]]'
                    ]
    
    for test_string in test_strings:
        read_dun_with_error(test_string, 4)
    
    #Test for invalid values
    test_strings = ['x : ?',
                    'x : ;',
                    'x : `',
                    ''
                    ]
    for test_string in test_strings:
        read_dun_with_error(test_string, 5)
        
    #Test depth 
    test_strings = ['x : [a : [g : [j : 0, k : 0] ], b : [g : [j : 0, k : 0] ]],',
                    'x : [0, 1]'
                    ]
    
    for test_string in test_strings:
        read_dun_with_error(test_string, 6, min_depth=2, max_depth=3)