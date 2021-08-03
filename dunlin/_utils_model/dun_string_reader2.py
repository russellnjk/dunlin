import re
from pathlib import Path

###############################################################################
#Non-Standard Imports
###############################################################################
try:
    from  .base_error  import DunlinBaseError
    from  .custom_eval import safe_eval as eval
except Exception as e:
    if Path.cwd() == Path(__file__).parent:
        from  base_error  import DunlinBaseError
        from  custom_eval import safe_eval  as eval
    else:
        raise e

###############################################################################
#Key-Value Parser for CLEANED DN Strings
###############################################################################
def read_dun(string, min_depth=0, max_depth=10):
    try:
        result = _read_dun(string, min_depth, max_depth)
    except DunlinSyntaxError as e:
        raise DunlinSyntaxError.merge(e, f'Error in string: {string}')
    except Exception as e:
        raise e
    
    if type(result) == dict:
        return result
    else:
        raise DunlinSyntaxError.top()


def _read_dun(string, min_depth=0, max_depth=10):
    chunks  = string.split('[')
    nest    = Nest()
    deepest = 0
    # print(chunks)
    # print()
    
    for i, chunk in enumerate(chunks):
        if i:
            nest    = nest.increase_depth()
            deepest = max(deepest, nest.depth)
        
        if nest.depth > max_depth:
            raise DunlinSyntaxError.depth('max')
            
        chunk = chunk.strip()
        # print('chunk', chunk)
        
        if chunk:
            if chunk[-1] not in [',', ':'] and i < len(chunks) - 1:
                raise DunlinSyntaxError.outside()
                
            sub_chunks = chunk.split(']')
            for ii, sub_chunk in enumerate(sub_chunks):
                if ii:
                    nest = nest.decrease_depth()
                
                sub_chunk = sub_chunk.strip()
                
                if sub_chunk:
                    if sub_chunk[0] not in [',', '*'] and ii:
                        raise DunlinSyntaxError.outside('aft')
                    
                    mul, data, last = read_sub_chunk(sub_chunk)
                else:
                    mul, data, last = None, None, None
                # print('nest', nest)
                # print('data', data)
                # print('last', last)
                
                if mul is not None:
                    nest.repeat_last_list(mul)
                if data is not None:
                    nest.update(data, last)
                    
    #     print('top view', nest.get_top())
    #     print('curr view', nest)

    #     print()
    
    # print(nest.depth)
    if nest.depth:
        raise DunlinSyntaxError.bracket('close')
    elif deepest < min_depth:
        raise DunlinSyntaxError.depth('min')
    return nest.release()
    
def is_repeat_list(data):
    try:
        return data[0][0] == '*'
    except:
        return False
    
def read_sub_chunk(sub_chunk):
    # print('sub', repr(sub_chunk))
    
    data_ = []
    count = 0
    for s in sub_chunk.strip().split(','):
        s_ = s.strip()
        if s_:
            data_.append(s_)
        else:
            count += 1
    
    if not len(data_):
        return None, None, None
    '''
    Check for 3 things:
        1. If there is a multiplier
        2. If the data is a list or dict
        3. If there is a "lone key" in the event that the data is a dict
    '''
    multiplier = None
    data       = None
    last       = None
    
    #Check for multiplier
    if data_[0][0] == '*':
        multiplier = eval(data_[0][1:])
        data_      = data_[1:]
    
    for i, datum in enumerate(data_, 0):
        temp = datum.split(':')
        if len(temp) == 1:
            dtype_ = list
            
        elif len(temp) == 2:
            dtype_ = dict
        else:
            raise Exception()
        
        if i == 0:
            dtype = dtype_
            data  = dtype()
        elif dtype != dtype_:
            raise DunlinSyntaxError.inconsistent()
        
        if dtype_ == list:
            value = read_value(temp[0])
            data.append(value)
        else:
            key, value = temp
            key   = read_value(key)
            value = read_value(value)
            if value is not None:
                data[key] = value
            else:
                last = key
                if i < len(data_) - 1:
                    raise DunlinSyntaxError.value(value)

    return multiplier, data, last

def read_value(x):
    try:
        return float(x)
    except:
        pass
    
    x_ = x.strip()
    if not x_:
        return None
    
    #Check for illegal characters including double underscore
    illegals = [':', ';', '?', '%', '$', 
                '#', '@', '!', '`', '~', '&', 
                '{', '}', '|', '\\', '__', 'None', ' '
                ]#The first backslash is an escape character!

    for s in  illegals:
        if s in x_:
            raise DunlinSyntaxError.value(x)
    
    try:
        return eval(x_)
    except:
        return x_

###############################################################################
#Nest Class
###############################################################################
class Nest():
    def __init__(self, _depth=0, _parent=None, _index=None):
        self.data   = None
        self.depth  = _depth
        self.parent = _parent
        self.index  = _index
        self.nest   = []
        self.key    = None
        
    def update(self, new_data, new_key):
        if self.data == None:
            self.data = new_data
            
        elif type(self.data) == dict and type(new_data) == dict:
            self.data.update(new_data)
            
        elif type(self.data) == list and type(new_data) == list:
            self.data += new_data
            
        else:
            raise DunlinSyntaxError.inconsistent()
            
        self.key = new_key
        
    def repeat_last_list(self, mul):
        if type(self.data) == list:
            self.data = self.data*mul
        else:
            last_key, last_value = list(self.data.items())[-1]
            if type(last_value) == list:
                self.data[last_key] = last_value*mul
            else:
                raise DunlinSyntaxError.value(last_value)
        
    def increase_depth(self):
        key = self.key
        
        if type(self.data) == dict:
            if key is None:
                raise Exception()
            new = Nest(_depth=self.depth+1, 
                       _parent=self, 
                       _index=key
                       )
            
            self.data[key] = new
            
        elif type(self.data) == list:
            if key is not None:
                raise Exception()
            new = Nest(_depth=self.depth+1, 
                       _parent=self, 
                       _index=len(self.data)
                       )
            
            self.data.append(new)
            
        elif self.data is None and key is None:
            new = Nest(_depth=self.depth+1, 
                       _parent=self, 
                       _index=0
                       )
            self.data = [new]
        else:
            raise DunlinSyntaxError.inconsistent()
        
        self.key = None
        return new
    
    def decrease_depth(self):
        if self.parent is None:
            raise DunlinSyntaxError.bracket('open')
        if self.parent.data is None:
            raise DunlinSyntaxError.bracket('open')
        else:
            self.parent.data[self.index] = self.release()
            return self.parent
    
    def release(self):
        if self.data is None:
            raise DunlinSyntaxError.value('Blank data.')
        return self.data
    
    def get_top(self):
        r = self
        while getattr(r, 'parent') is not None:
            r = r.parent
        return r
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def __repr__(self):
        return f'Nest({self.data})'
    
    def __str__(self):
        return self.__repr__()
    
###############################################################################
#Key-Value Parser for CLEANED PY Strings
###############################################################################
def format_indent(string):
    try:
        name, code = string.split(':', 1)
    except:
        raise DunlinSyntaxError.py(string)
        
    name = name.strip()
    
    split = code.strip().split('\n', 1)
    
    #Single line  of code
    if len(split) == 1:
        return name, '\t' + split[0].strip()
    
    #All other cases
    line0, body = split
    first_indent = re.search('\s*', split[1])[0]
    line0        = first_indent + line0.strip()
    formatted    = line0 + '\n' + body
    
    return name, formatted
    
###############################################################################
#Dunlin Exceptions
###############################################################################
class DunlinSyntaxError(SyntaxError, DunlinBaseError):
    @classmethod
    def inconsistent(cls):
        return cls.raise_template('Inconsistent data type.', 1)
    
    @classmethod
    def outside(cls, pos='bef'):
        details = 'Values before bracket.' if pos == 'bef' else 'Values after bracket.'
        return cls.raise_template(details, 2)
    
    @classmethod
    def top(cls):
        return cls.raise_template('Top level must be dict-like.', 3)
    
    @classmethod
    def bracket(cls, miss='open'):
        if miss == 'open':
            details = 'Detected an unexpected closing bracket or missing opening bracket.'
        else:
            details = 'Detected an unexpected opening bracket or missing closing bracket.'
        return cls.raise_template(details, 4)
    
    @classmethod
    def value(cls, x):
        return cls.raise_template(f'Invalid value: {repr(x)}.', 5)
    
    @classmethod
    def depth(cls, bnd='min'):
        if bnd == 'min':
            details = 'Minimum depth not satisfied.'
        else:
            details = 'Maximum depth exceeded.'
        return cls.raise_template(details, 6)
    
    @classmethod
    def py(cls, string):
        return cls.raise_template(f'Missing code or invalid subsection format: {string}', 7)
    
if __name__ == '__main__':
    s = 'x : 0, y : 1, z : [a : [j: 2, k: 2], b : 3, c: [4, 5,]*2, d:6,], h: [[5, [5]], [5, 5]]'
    
    r = read_dun(s)
    print(r)
    
    ###############################################################################
    #DN Subsection Code Strings without Errors
    ###############################################################################
    def test_string_without_error(test_string, answer, **kwargs):
        r = read_dun(test_string, **kwargs)
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
                      {'x': [0, 1, 0, 1], 'y': [0, 1, 2]}],
                    ['x : [0, 1], y : [[0, 1, 2], [0, 1, 2]]',
                      {'x': [0, 1], 'y': [[0, 1, 2], [0, 1, 2]]}],
                    ['x : [0, 1], y : [[0, 1, 2], [0, 1, 2]], z : [a : 3, b : 4]',
                      {'x': [0, 1], 'y': [[0, 1, 2], [0, 1, 2]], 'z': {'a': 3, 'b': 4}}],
                    ['x : 0, y : 1, z : [a : [j: 2, k: 2], b : 3, c: [4, 5,]*2, d:6,], h: [[5, [5]], [5, 5]]',
                      {'x': 0.0, 'y': 1.0, 'z': {'a': {'j': 2.0, 'k': 2.0}, 'b': 3.0, 'c': [4.0, 5.0, 4.0, 5.0], 'd': 6.0}, 'h': [[5.0, [5.0]], [5.0, 5.0]]}],
                    ]
    
    for test_string, answer in test_strings:
        test_string_without_error(test_string, answer)
    
    ###############################################################################
    #DN Subsection Code Strings with Errors
    ###############################################################################
    def read_dun_with_error(test_string, num, **kwargs):
        expected = False
        try:
            read_dun(test_string, **kwargs)
        except DunlinSyntaxError as e:
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
    test_strings = ['x : [a : [g : [j : 0, k : [[0]]] ],',
                    'x : [0, 1]'
                    ]
    
    for test_string in test_strings:
        read_dun_with_error(test_string, 6, min_depth=2, max_depth=3)
        
    ###############################################################################
    #PY Subsection Code
    ###############################################################################
    test_strings = [['x : a = 1\n\tb = 1', '\ta = 1\n\tb = 1'],
                    ['x :  \n  a = 1\n\tb = 1', '\ta = 1\n\tb = 1'],
                    ['x:\n\tb=1', '\tb=1']
                    ]
    
    for test_string, answer in test_strings:
        _, r = format_indent(test_string)
        assert r == answer
    
    #No error testing as code validity is defaulted to the Python interpreter.
    test_strings = [['x : a : 1\n\tb : 1', '\ta : 1\n\tb : 1'],
                    ['x :  a : 1\n  b : 1', '  a : 1\n  b : 1'],
                    ['x:a:1 \n  b:1', '  a:1\n  b:1']
                    ]
    
    for test_string, answer in test_strings:
        _, r = format_indent(test_string)
        assert r == answer
    