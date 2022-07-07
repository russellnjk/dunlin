import pyrfc3339 as datetime

# illegals = [':', ';', '?', '%', '$', 
#             '#', '@', '!', '`', '~', '&', 
#             '{', '}', '|',  
#             '\\', '__',
#             ]
illegals = ['__']

def read_primitive(x):
    '''
    Reads a single value "x" which corresponds to one of the 4 cases:
        1. x is a number.
        2. x is a math expression.
        3. x is a datetime
        4. x is a Boolean
        5. x is a string
        
    Assumes that x has already been stripped.
    
    Notes
    -----
    Double underscores are illegal for safety reasons. To change set of illegal 
    characters, you can modify the global variable ```illegals```.
    '''
    
    #Case 1: x is a number
    try:
        return int(x)
    except:
        try:
            return float(x)
        except:
            pass
    
    #Case 2: x is a math expression
    if ismath(x):
        try:
            return eval(x, {}, {})
        except:
            raise DunValueError(x)
            
    #Case 3: x is a datetime 
    try:
        return datetime.parse(x)
    except:
        pass
    
    #Case 4: x is a boolean
    lower =  x.lower()
    if lower == 'true':
        return True
    elif lower == 'false':
        return False
        
    #Case 5: x is a string
    #Case 5a: x is a string with quotes.
    #Strip quotes ONCE and continue to next
    if x[0] in ['"', "'"]:
        if x[0] == x[-1]:
            x = x[1:-1]
        else:
            raise DunValueError(x)
    
    
    #Case 5b: x is a string without quotes
    #Check for illegal symbols
    if hasillegals(x):
        raise DunValueError(x)
        
    return x

def hasillegals(x):
    global illegals
    for s in  illegals:
        if s in x:
            return True
    return False
    
notnum = set('.+-*/%() \n\t\r')
math   = set('0123456789.+-*/%() \n\t\r')
def ismath(x):
    '''Tests if a string is a math expression. Assumes the string is not a number.
    '''
    global notnum
    global math

    return all([char in math for char in x]) and not all([char in notnum for char in x])

class DunValueError(Exception):
    def __init__(self, x):
        super().__init__(f'Invalid value: {repr(x)}.')

