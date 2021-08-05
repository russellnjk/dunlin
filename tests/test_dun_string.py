import addpath
import dunlin                                as dn
import dunlin._utils_model.dun_string_reader as dsr

if __name__ == '__main__':
    #Test preprocess_string
    print('Test preprocess_string')
    new = dsr.preprocess_string('a')
    assert new == 'a'
    
    new = dsr.preprocess_string('a,')
    assert new == 'a'
    
    try:
        new = dsr.preprocess_string('a,,')
    except Exception as e:
        assert e.num == 2
    
    try:
        new = dsr.preprocess_string(',')
    except Exception as e:
        assert e.num == 7
        
    assert new == 'a'
    print()
    #Test append_chunk
    print('Test append_chunk')
    
    
    print()
    #Test _read_dun without parsing
    print('Test _read_dun without parsing')
    rf = lambda x: x
    
    s = 'a: 0, b: 1'
    r = dsr._read_dun(s, rf)
    assert r == ['a: 0', 'b: 1']
    print(r)
    
    s = 'a: [0], b: 1'
    r = dsr._read_dun(s, rf)
    assert r == ['a:', ['0'], 'b: 1']
    print(r)
    
    s = 'a: [[0]], b: 1'
    r = dsr._read_dun(s, rf)
    assert r == ['a:', [['0']], 'b: 1']
    print(r)
    
    s = 'a: [[0], 10], b: 1'
    r = dsr._read_dun(s, rf)
    assert r == ['a:', [['0'], '10'], 'b: 1']
    print(r)
    
    s = 'a: [[0]*2, 10], b: 1'
    r = dsr._read_dun(s, rf)
    assert r == ['a:', [['0'], '*2', '10'], 'b: 1']
    print(r)
    
    s = 'a: [[0]*2, [10]*2], b: 1'
    r = dsr._read_dun(s, rf)
    assert r == ['a:', [['0'], '*2', ['10'], '*2'], 'b: 1']
    print(r)
    
    s = 'a: [x: 0], b: 1'
    r = dsr._read_dun(s, rf)
    assert r == ['a:', ['x: 0'], 'b: 1']
    print(r)
    
    s = 'a: [x: 0], b: [1]'
    r = dsr._read_dun(s, rf)
    assert r == ['a:', ['x: 0'], 'b:', ['1']]
    print(r)
    
    s = 'a: [x: 0], b: [y: 1, z: [2]]'
    r = dsr._read_dun(s, rf)
    assert r == ['a:', ['x: 0'], 'b:', ['y: 1', 'z:', ['2']]]
    print(r)
    
    s = 'a: [x: 0], b: [y: 1, z: [[2]*3, 4]]'
    r = dsr._read_dun(s, rf)
    assert r == ['a:', ['x: 0'], 'b:', ['y: 1', 'z:', [['2'], '*3', '4']]]
    print(r)
    
    s = 'a: [x: 0]*2'
    r = dsr._read_dun(s, rf)
    assert r == ['a:', ['x: 0'], '*2']
    print(r)
    
    print()
    #Test trailing comma
    print('Test trailing comma')
    s = 'a: [x: 0]*2,'
    r = dsr._read_dun(s, rf)
    assert r == ['a:', ['x: 0'], '*2']
    print(r)
    
    print()
    #Test value before bracket
    print('Test value before bracket')
    try:
        s = 'a [x: 0]*2'
        r = dsr._read_dun(s, rf)
    except Exception as e:
        print('Caught:', e)
        assert e.num == 1
    else:
        assert False
    
    print()
    #Test value after bracket
    print('Test value after bracket')
    try:
        s = 'a: [x: 0] 2, b: 3'
        r = dsr._read_dun(s, rf)
    except Exception as e:
        print('Caught:', e)
        assert e.num == 1
    else:
        assert False
    
    try:
        s = 'a: [x: 0] 2'
        r = dsr._read_dun(s, rf)
    except Exception as e:
        print('Caught:', e)
        assert e.num == 1
    else:
        assert False
    
    print()
    #Test unexpected delimiters
    print('Test unexpected delimiters')
    test_strings = [',a', '[, a]', '0,,']
    for s in test_strings:
        try:
            r = dsr._read_dun(s, rf)
        except Exception as e:
            print('Caught: ', e)
            assert e.num == 2
        else:
            assert False
    
    print()
    #Test mismatched brackets
    print('Test mismatched brackets')
    test_strings = [']', 'a: [0]]', '[', 'a: [0']
    for s in test_strings:
        try:
            r = dsr._read_dun(s, rf)
            print(r)
        except Exception as e:
            print('Caught: ', e)
            assert e.num == 6
        else:
            assert False
    
    '''
    By now every element is either a list or a non-blank string!
    '''
    print()
    #Test read_value
    print('Test read_value')
    s = ' 4'
    r = dsr.read_value(s)
    assert r == 4

    '''
    A "flat" list can only contain:
        1. Non-blank strings 
            1. Raise error if colon (:) in string
            2. Check if it a repeat
                1. Check that the previous element is a list
                2. Raise error otherwise
            3. Call read_value and append otherwise
        2. Parsed lists and parsed dicts
            1. Append to result
    '''
    print()
    #Test read_list
    print('Test read_list')
    
    flat = ['a', 'b', ['c', 'd']]
    r = dsr.read_list(flat)
    assert r == ['a', 'b', ['c', 'd']]
    print(r)
    
    flat = ['a', 'b', ['c', 'd'], '*2']
    r = dsr.read_list(flat)
    assert r == ['a', 'b', ['c', 'd', 'c', 'd']]
    print(r)
    
    flat = ['a', 'b', {'c': 0, 'd': 1}]
    r = dsr.read_list(flat)
    assert r == ['a', 'b', {'c': 0, 'd': 1}]
    print(r)
    
    print()
    #Test inconsistent list
    print('Test inconsistent list')
    try:
        flat = ['a', 'b:', ['c', 'd'], '*2']
        r = dsr.read_list(flat)
    except Exception as e:
        print('Caught:', e)
        assert True
    else:
        assert False
    
    print()
    #Test repeat followed after a non-list
    print('Test repeat followed after a non-list')
    flats = [ ['a', 'b', {'c': 0, 'd': 1}, '*2'],
              ['a', 'b', 'c', '*2']
              ]
    for flat in flats:
        try:
            r = dsr.read_list(flat)
        except Exception as e:
            print('Caught:', e)
            assert True
        else:
            assert False
    
    '''
    A "flat" dict can only contain:
        1. Non-blank strings 
            1. Split into a key/value pair.
            2. The string is a key. Save it for the next value.
            3. A multiplier
        2. Parsed lists and parsed dicts
            1. Check the if the previous element contained a key without a value.
    '''
    print()
    #Test read_dict
    print('Test read_dict')
    flat = ['a: 0', 'b: ', {'x': 1, 'y': 2}, 'c: 3']
    r = dsr.read_dict(flat)
    print(r)
    assert r == {'a': 0, 'b': {'x': 1, 'y': 2}, 'c': 3}
    
    flat = ['a: 0', 'b: ', [1, 2], 'c: 3']
    r = dsr.read_dict(flat)
    print(r)
    assert r == {'a': 0, 'b': [1, 2], 'c': 3}
    
    flat = ['a: 0', 'b: ', [1, 2], '*2']
    r = dsr.read_dict(flat)
    print(r)
    assert r == {'a': 0, 'b': [1, 2, 1, 2]}
    
    print()
    #Test inconsistent dict
    print('Test inconsistent dict')
    try:
        flat = ['a: ', 'b: ', {'x': 1, 'y': 2}, 'c: 3']
        r = dsr.read_dict(flat)
    except Exception as e:
        print('Caught:', e)
        assert True
    else:
        assert False
    
    print()
    #Test missing value
    print('Test missing value')
    try:
        flat = ['a: 0', 'b: ',]
        r = dsr.read_dict(flat)
    except Exception as e:
        print('Caught:', e)
        assert True
    else:
        assert False
    
    #Test missing/blnk key
    print('Test missing/blank key')
    try:
        flat = ['a: 0', ' : ', ]
        r = dsr.read_dict(flat)
    except Exception as e:
        print('Caught:', e)
        assert True
    else:
        assert False
        
    print()
    #Test _read_dun with parsing
    print('Test _read_dun')
    rf = lambda x: x
    rf = dsr.read_flat
    
    s = 'a: 0, b: 1'
    r = dsr._read_dun(s, rf)
    assert r == {'a': 0, 'b': 1}
    print(r)
    
    s = 'a: [0], b: 1'
    r = dsr._read_dun(s, rf)
    assert r == {'a': [0], 'b': 1}
    print(r)
    
    s = 'a: [[0]], b: 1'
    r = dsr._read_dun(s, rf)
    assert r == {'a': [[0]], 'b': 1}
    print(r)
    
    s = 'a: [[0], 10], b: 1'
    r = dsr._read_dun(s, rf)
    assert r == {'a': [[0], 10], 'b': 1}
    print(r)
    
    s = 'a: [[0]*2, 10], b: 1'
    r = dsr._read_dun(s, rf)
    assert r == {'a': [[0, 0], 10], 'b': 1}
    print(r)
    
    s = 'a: [[0]*2, [10]*2], b: 1'
    r = dsr._read_dun(s, rf)
    assert r == {'a': [[0, 0], [10, 10]], 'b': 1}
    print(r)
    
    s = 'a: [x: 0], b: 1'
    r = dsr._read_dun(s, rf)
    assert r == {'a': {'x': 0}, 'b':1}
    print(r)
    
    s = 'a: [x: 0], b: [1]'
    r = dsr._read_dun(s, rf)
    assert r == {'a': {'x': 0}, 'b': [1]}
    print(r)
    
    s = 'a: [x: 0], b: [y: 1, z: [2]]'
    r = dsr._read_dun(s, rf)
    assert r == {'a': {'x': 0}, 'b': {'y': 1, 'z': [2]}}
    print(r)
    
    s = 'a: [x: 0], b: [y: 1, z: [[2]*3, 4]]'
    r = dsr._read_dun(s, rf)
    assert r == {'a': {'x': 0}, 'b': {'y': 1, 'z': [[2, 2, 2], 4]}}
    print(r)
    
    s = 'a: [0]*2'
    r = dsr._read_dun(s, rf)
    assert r == {'a': [0, 0]}
    print(r)
    
    s = 'a: [b: [0]*2]'
    r = dsr._read_dun(s, rf)
    assert r == {'a': {'b': [0, 0]}}
    print(r)
    
    print()
    s = 'a : [[0: 1, 2: [3, 4]], [b, c], [d, [e]], f], g: [h], i: [j, k, l, m, n, [o, p, q] ]'
    r = dsr._read_dun(s, rf)
    print(r)
    
    print()
    #Test depth control
    print('Test depth control')
    s = 'a: 1'
    try:
        r = dsr._read_dun(s, rf, min_depth=1)
    except Exception as e:
        print('Caught:', e)
        assert e.num == 9
    else:
        assert False
    
    s = 'a: [[1]]'
    try:
        r = dsr._read_dun(s, rf, max_depth=1)
    except Exception as e:
        print('Caught:', e)
        assert e.num == 9
    else:
        assert False
    
    