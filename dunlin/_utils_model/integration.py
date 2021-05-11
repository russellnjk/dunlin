import numpy             as np
from   numba             import jit
from   scipy.integrate   import solve_ivp
from   scipy.optimize    import root

###############################################################################
#Piecewise Integration
###############################################################################    
def piecewise_integrate(function, tspan, init, params, inputs, scenario, modify=None, args=(), **solver_args):
    solver_args_ = solver_args if solver_args else {'method': 'LSODA'}
    args_        = tuple(args)
    init_, y_args = int_args_helper(function, init, params, inputs, scenario, 0, modify, args_)
    
    tspan_  = tspan[0]
    sol     = solve_ivp(function, (tspan_[0], tspan_[-1]), init_, t_eval=tspan_, args=y_args, **solver_args_)
    y_model = sol.y
    t_model = tspan[0]
    
    for segment in range(1, len(tspan)):
        init_, y_args = int_args_helper(function, y_model[:,-1], params, inputs, scenario, segment, modify, args_)
            
        tspan_  = tspan[segment]
        sol     = solve_ivp(function, (tspan_[0], tspan_[-1]), init_, t_eval=tspan_, args=y_args, **solver_args_)
        y_      = sol.y  

        y_model = np.concatenate((y_model, y_    ), axis=1) 
        t_model = np.concatenate((t_model, tspan_), axis=0)
        
    return y_model, t_model

def int_args_helper(function, init, params, inputs, scenario, segment, modify=None, args=()):
    if inputs is None:
        if modify:
            init_, params_ = modify(function=function, init=init, params=params, scenario=scenario, segment=segment)
        else:
            init_, params_ = init, params
        y_args = tuple([params_]) + args
    else:
        if modify:
            init_, params_, inputs_ = modify(function=function, init=init, params=params, inputs=inputs[segment], scenario=scenario, segment=segment)
        else:
            init_, params_, inputs_ = init, params, inputs[segment]
        y_args = tuple([params_, inputs_]) + args

    return init_, y_args    

###############################################################################
#Steady State 
###############################################################################    
def get_steady_state(function, init, params, input, **kwargs):
    
    if input is None:
        f = lambda y: function(0, y, params)
    else:
        f = lambda y: function(0, y, params, input)
    
    new_init = root(f, init, **kwargs)
    return new_init
    
###############################################################################
#Templates
###############################################################################    
modify_with_inputs = '''
def modify_with_inputs(function, init, params, inputs, scenario, segment):
    """
    Return a new np.array of initial values. For safety, DO NOT MODIFY IN PLACE. 
    """
    new_init   = init.copy()
    new_params = params.copy()
    new_inputs = inputs.copy()
    
    return new_init, new_params, new_inputs
'''

modify_without_inputs = '''
def modify_without_inputs(function, init, params, scenario, segment):
    """
    Return a new np.array of initial values. For safety, DO NOT MODIFY IN PLACE.
    """
    new_init   = init.copy()
    new_params = params.copy()
    new_inputs = inputs.copy()
    
    return new_init, new_params
'''

if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    plt.close('all')
    
    #Preprocessing
    @jit(nopython=True)
    def model(t, y, params, inputs):
        a = y[0]
        b = y[1]
        
        q = params[0]
        r = params[1]
        
        u = inputs[0]
        v = inputs[1]
        
        da = -q*a + u
        db = -q*b + v + r*a
        
        return np.array([da, db])
    
        
    tseg1 = np.linspace(0,   40, 21)
    tseg2 = np.linspace(40,  80, 21)
    tseg3 = np.linspace(80, 120, 21)
    
    tspan = [tseg1, tseg2, tseg3]
    
    iseg1 = np.array([0,   1])
    iseg2 = np.array([1,   0])
    iseg3 = np.array([0, 0.5])
    
    inputs = [iseg1, iseg2, iseg3]
    
    scenario = 1
    
    init   = np.array([10,  0])
    params = np.array([0.2, 0.5])
    
    #Test integration
    y_model, t_model = piecewise_integrate(model, tspan, init, params, inputs, scenario)
    
    fig = plt.figure()
    AX  = [fig.add_subplot(2, 1, i+1) for i in range(2)]
    
    for i, ax in enumerate(AX):
        ax.plot(t_model, y_model[i])
    
    assert y_model.shape == (2, 63)
    
    #Test modifier
    def modify1(init, params, inputs, scenario, segment):
        new_init   = init.copy()
        new_params = params.copy()
        new_inputs = inputs.copy()
        
        new_init[0] *= 4
        # print(new_inputs)
        return new_init, new_params, new_inputs
    
    y_model, t_model = piecewise_integrate(model, tspan, init, params, inputs, scenario, modify=modify1)
    
    fig = plt.figure()
    AX  = [fig.add_subplot(2, 1, i+1) for i in range(2)]
    
    for i, ax in enumerate(AX):
        ax.plot(t_model, y_model[i])
    
    assert y_model.shape == (2, 63)
    
    #Test steady-state finder
    ss = get_steady_state(model, np.array([10, 10]), np.array([0.2, 0.5]), np.array([1, 1]))
    ss = ss.x
    
    assert all(np.isclose(ss, [5, 17.5], atol=1e-3))
    