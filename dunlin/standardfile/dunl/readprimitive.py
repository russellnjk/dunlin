# import pyrfc3339 as datetime
from datetime  import datetime
from pyrfc3339 import parse as parse_datetime
from numbers   import Number

illegals = {'__'}

def read_primitive(x: str) -> Number|str|bool|datetime:
    '''
    Reads a single value "x" which corresponds to one of the 4 cases:
        1. x is a number/math expression
        3. x is a datetime
        4. x is a Boolean
        5. x is a string
        
    Assumes that x has already been stripped.
    
    Notes
    -----
    Double underscores are illegal for safety reasons. To change set of illegal 
    characters, you can modify the global variable ```illegals```.
    
    '''
    
    x = x.strip()
    
    #Raise an exception in these cases
    if not x:
        msg = 'Encountered a blank value when reading a string.'
        raise ValueError(msg)
    elif '__' in x:
        msg  = f'Error reading the following: {x}'
        msg += 'Double underscores are not allowed.'
        raise ValueError(msg)
    
    #Case 1: x is a boolean
    elif x == 'True':
        return True
    elif x == 'False':
        return False
    
    #Case 2: x is a number or mathematical expression
    elif ismath(x):
        return eval(x, {}, {})
            
    #Case 3: x is a datetime 
    try:
        return parse_datetime(x)
    except:
        pass
    
    #Case 4: x is a string
    if len(x) == 1:
        return x
    elif x[0] == x[-1] and x[0] in '\'"':
        return x[1:-1]
    else:
        return x
    
notnum = set('.+-*/%() \n\t\r')
math   = set('0123456789.+-*/%() \n\t\r')
def ismath(x: str) -> bool:
    '''Tests if a string is a math expression. Assumes the string is not a number.
    '''
    global notnum
    global math
    
    has_numbers = False
    for char in x:
        if char not in math:
            return False
    
        elif char not in notnum:
            has_numbers = True
            
    if has_numbers:
        return True
    else:
        return False
    

