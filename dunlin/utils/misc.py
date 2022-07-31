import re
from numbers import Number

###############################################################################
#Submodels, Differentials and Namespaces
###############################################################################
def undot(x, ignore=lambda x: x[:2] == '__'):
    if isstrlike(x):
        def repl(x):
            variable = x[0]
            
            if ignore(variable):
                return variable
            else:
                return variable.replace('.', '__dot__')
        
        return re.sub('[a-zA-Z_]+[\w_]*\.', repl, x)
    else:
        return [undot(i) for i in x]
    
def dot(x):
    if isstrlike(x):
        return x.replace('__dot__', '.')
        # def repl(x):
        #     return x[0].replace('__dot__', '.')
        
        # return re.sub('[A-Za-z]__dot__[A-Za-z]', repl, x)
    else:
        return [dot(i) for i in x]
    
def sub(*x):
    return '.'.join(x)

def unsub(x):
    return tuple(x.split('.'))

def issub(x):
    if '.' in x or '__dot__' in x:
        return True
    else:
        return False

def diff(x):
    if type(x) == str:
        return 'd_' + x
    else:
        return [diff(i) for i in x]

def undiff(x):
    if type(x) == str:
        return x.split('d_', 1)[1]
    else:
        return [undiff(i) for i in x]

def isdiff(x):
    if not isstrlike(x):
        raise TypeError(f'Expected a string. Received {type(x)}')
    
    if len(x) < 2:
        return False
    else:
        return x[:2] == 'd_'

def check_not_diff(x):
    if islistlike(x) or isdictlike(x):
        temp = [check_not_diff(i) for i in x]
        return any(temp)
    
    if isdiff(x):
        raise NameError(f'{x} is a differential and is reserved.')

def is_valid_name(x):
    try:
        check_valid_name(x)
        return True
    except:
        return False

reserved = ['states', 'parameters', 'time', 'posterior', 'context', 'priors', 
            'objective', 'all',
            'True', 'False'
            ]

def check_valid_name(x, allow_reserved=False):
    if not x:
        raise NameError('Name cannot be blank.')
    
    if type(x) != str:
        raise TypeError('Variable name must be a string.')
        
    check_not_diff(x)
    
    if x in reserved and not allow_reserved:
        raise NameError(f'Reserved namespace {x}')
    
    if any([i.isspace() for i in x]):
        raise NameError(f'Detected whitespace in name : {repr(x)}')
    
    if x[0] == '_':
        raise NameError('Name cannot start with an underscore.')
    
    if '__' in x:
        raise NameError('Name cannot contain double underscores.')
    
###############################################################################
#Other Type Checking/Conversion
###############################################################################
def try2num(x):
    try:
        return str2num(x)
    except:
        return x
    
def str2num(x):
    try:
        fx = float(x)
    except:
        raise ValueError(f'Could not convert {x} into a number.')
    
    try:
        ix = int(x)
        
        if ix == fx:
            return ix
        else:
            return fx
    except:
        return fx
    
# def val2list(x):
#     if isnum(x) or isstrlike(x):
#         return [x]
#     else:
#         return list(x)

# def val2dictoflist(x):
#     if isnum(x) or isstrlike(x):
#         return {0: x}
#     elif islistlike(x):
#         return dict(enumerate(x))
#     else:
#         return dict(x)  

def isint(x):
    try:
        x_ = int(x)
        return x_ == x
    except:
        return False
    
def isnum(x):
    return isinstance(x, Number)

def strisnum(x):
    if type(x) != str:
        raise TypeError('Expected a string.')
    try:
        float(x)
        return True
    except:
        return False
    
def isdictlike(x):
    return hasattr(x, 'items')

def islistlike(x):
    result = hasattr(x, '__iter__') and not hasattr(x, 'items') and not hasattr(x, 'join')
    return result

def istuplelike(x):
    if islistlike(x) and not hasattr(x, 'append'):
        return True
    else:
        return False
    
def isstrlike(x):
    try:
        x + 's'
    except:
        return False
    
    if hasattr(x, '__iter__') and hasattr(x, 'join'):
        return True
    else:
        return False

notnum = set('.+-*/%() \n\t\r')
math   = set('0123456789.+-*/%() \n\t\r')
def ismath(x):
    '''Tests if a string is a math expression. Assumes the string is not a number.
    '''
    global notnum
    global math

    return all([char in math for char in x]) and not all([char in notnum for char in x])


def split_functionlike(x, allow_num=True):
    #For splitting f(x...) into f, (x...)
    
    if type(x) != str:
        return [split_functionlike(i) for i in x]
    
    found = re.findall('^(\w[\w.]*)\((.*)\)$', x.strip())
    
    if not found:
        raise ValueError(f'Could not split {x} into function name and signature.')
        
    name, signature = found[0]
    
    args = []
    for v in signature.split(','):
        v = v.strip()
        if v.isidentifier():
            args.append(v)
        elif isnum(v) and allow_num:
            args.append(v)
        else:
            raise ValueError(f'Invalid variable name in function {x}: {v}')
    
    signature = ', '.join(args)
    return name, signature, args

###############################################################################
#Variable Extraction
###############################################################################
def get_namespace(eqn, allow_reserved=False, add_reserved=False):
    if not eqn:
        return set()
    
    if islistlike(eqn):
        result = set()
        for e in eqn:
            temp = get_namespace(e, allow_reserved, add_reserved)
            result.update(temp)
        return result
    else:
        pattern   = '[a-zA-Z_][a-zA-Z0-9_.]*|[0-9]e[0-9]'
        pattern   = '[a-zA-Z_][a-zA-Z0-9_.]*|[0-9]e-?[0-9]'
        found     = re.findall(pattern, eqn.strip()) 
        variables = set()
        
        for v in found:
            if strisnum(v):
                continue
            elif v in reserved:
                if allow_reserved:
                    if add_reserved:
                        variables.add(v)
                    else:
                        continue
                else:
                    msg = f'Detected reserved keyword in equation {eqn}: {v}'
                    raise NameError(msg)
            else:
                variables.add(v)
        
        return variables

###############################################################################
#Coding
###############################################################################
def T(n, s=''):
    return '\t'*n + s

###############################################################################
#Scenario and Variable comparison
###############################################################################
def compare_variables(v0, v1):
    def is_valid(v):
        return type(v) == str or isnum(v) or v is None
    
    if not is_valid(v0) or not is_valid(v1):
        msg = 'Could not compare variables {v0} and {v1} due to unexpected type.'
        raise TypeError(msg)
    elif v0 is None or v1 is None:
        return True
    else:
        return v0 == v1
    
def compare_scenarios(c0, c1):
    def is_valid(c):
        return type(c) in [tuple, str] or isnum(c) or c is None
    
    if not is_valid(c0) or not is_valid(c1):
        msg = f'Could not compare scenarios {c0} and {c1} due to unexpected type.'
        raise TypeError(msg)
    elif c0 is None or c1 is None:
        return True
    elif type(c0) == str or isnum(c0):
        return c0 == c1
    else:
        if len(c0) != len(c1):
            return False
        
        test = [i == ii or i is None or ii is None for i, ii in zip(c0, c1)]
        
        return all(test)
    
    
    