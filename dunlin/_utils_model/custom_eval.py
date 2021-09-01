import re

def safe_eval(string, patterns = ('__', '[A-Za-z]\.', 'lambda +[\w]'), locals=None):
    '''
    For safe evaluation of strings.
    
    :meta private:
    '''
    string
    
    for pattern in patterns:
        __r = re.search(pattern, string)

        if __r:
            msg = 'This string contains invalid characters and might be unsafe: \n{}'
            raise ValueError(msg.format(string))
    
    return eval(string, locals)

if __name__ == '__main__':
    
    #Test safety
    a = '[1, 2, 3]'
    b = '__import__.subprocess.get()'
    c = '1e-8'
    d = '1e8'
    e = 'with open(file, 'r') as file:'
    f = "q = (q.gi_frame.f_back.f_back.f_globals for _ in (1,)); builtins = [*q][0]['_' + '_builtins_' + '_']; builtins.print('Gotcha:', builtins.dir(builtins))"
    
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