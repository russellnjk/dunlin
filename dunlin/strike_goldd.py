from   sympy          import symbols
from   sympy.matrices import Matrix
import textwrap as tw

###############################################################################
#Non-Standard Imports
###############################################################################
# import dunlin.model              as dml
import dunlin._strike_goldd_algo as sga

###############################################################################
#Wrapper for Algorithm
###############################################################################

###############################################################################
#Symbolic Generator
###############################################################################
def convert2symbolic(model, **kwargs):
    pstr     = model.get_param_names()
    xstr     = model.get_state_names()
    p        = symbols(pstr)
    x        = symbols(xstr)
    symbolic = dict(zip(pstr+xstr, [*p, *x]))
    
    #Collate arguments
    model_ics = model.states.iloc[0].to_dict()
    kwargs_   = getattr(model, 'strike_goldd_args', {})
    kwargs_   = {**model_ics, **kwargs_, **kwargs}

    def test_h(h):
        if len(h) and all([i in x for i in h]):
            return True
        return False
    
    def test_u(u):
        if len(u):
            if all([i in p and u[i] >= 0 for i in u]):
                return True
            return False
        return True
    
    def test_ics(ics):
        if len(ics) == len(x) and all([i in x for i in ics]):
            return True
        return False
    
    #Extract arguments
    h      = _get_args('observed', kwargs_, symbolic, test_h  )
    u      = _get_args('inputs',   kwargs_, symbolic, test_u  )
    ics    = _get_args('init',     kwargs_, symbolic, test_ics)
    decomp = _get_decomp(kwargs_, symbolic)
        
    eqns = tw.dedent(model._eqns)
    symbolic['states'] = x
    symbolic['params'] = p
    
    exec(eqns, None, symbolic)
    
    f = Matrix([symbolic[f'd_{i}'] for i in x])
    
    #Substitute known params
    nominal   = model.params.iloc[0].to_dict()
    unknown_p = []
    to_sub    = {}
    
    for param in p:
        if param in kwargs_['unknown']:
            unknown_p.append(param)
        else:
            to_sub[param] = nominal[str(param)]
            
    unknown_p = Matrix(unknown_p)
    f_sub     = f.subs(to_sub)
    
    #Create the dictionary
    result = {'h': Matrix(h), 'x': Matrix(x), 'p': unknown_p, 'f': f_sub, 'u': u, 'ics': ics, 'decomp' : decomp}

    return symbolic, result
    
def _get_args(arg, kwargs, symbolic, test):
    arg_ = kwargs.get(arg, {})
    
    if type(arg_) == dict:
        arg_ = {symbolic[k]: v for k, v in arg_.items()} 
    else:
        arg_ = [symbolic[k] for k in arg_]
    
    if not test(arg_):
        raise ValueError(f'{arg} is invalid.')
    
    return arg_
    
def _get_decomp(kwargs, symbolic):
    decomp_ = kwargs.get('decomp', [])
    decomp  = []
    for lst in decomp_:
        if not lst:
            raise ValueError('Blank decomposition group.')
        
        try:
            lst_ = [symbolic[i] for i in lst]
        except KeyError as e:
            raise KeyError(f'list in decomp contains an unexpected state: {e.args}.')
        
        decomp.append(lst_)
    
    return decomp
    