import re

#'[A-Za-z]'
def safe_eval(string, pattern = '__', replace=None):
    '''
    For safe evaluation of strings.
    
    :meta private:
    '''
    r = string
    
    if replace:
        for word in replace:
            r = r.replace(word, '')
    
    r = re.findall(pattern, r)
    
    if r:
        msg = 'This string contains invalid characters and might be unsafe: \n{}'
        raise ValueError(msg.format(string))
    else:
        return eval(string, {'__builtins__': None}, {})

if __name__ == '__main__':
    
    #Test safety
    a = '[1, 2, 3]'
    b = '__import__.subprocess.get()'
    c = '1e-8'
    d = '1e8'
    e = 'with open(file, 'r') as file:'
    
    def tester(s):
        try:
            safe_eval(s)
            return 1
        except:
            return 0
    
    r = tester(a)
    assert r == 1
    
    r = tester(b)
    assert r == 0
    
    r = tester(c)
    assert r == 1
    
    r = tester(d)
    assert r == 1