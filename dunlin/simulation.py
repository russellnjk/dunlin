import numpy   as np
import pandas  as pd
from   pathlib import Path

###############################################################################
#Non-Standard Imports
###############################################################################
try:
    import dunlin.model_handler            as mh
    import dunlin._utils_model.integration as itg
    import dunlin._utils_plot.utils_plot   as utp
except Exception as e:
    if Path.cwd() == Path(__file__).parent:
        import model_handler            as mh
        import _utils_model.integration as itg
        import _utils_plot.utils_plot   as utp
    else:
        raise e

###############################################################################
#Globals
###############################################################################
colors        = utp.colors
palette_types = utp.palette_types
fs            = utp.fs

###############################################################################
#High-Level Functions
###############################################################################
def integrate_and_plot(sim_args, plot_index, AX=None, **kwargs):
    
    simulation_results = integrate_models(sim_args)
    figs, AX           = plot_simulation_results(plot_index, simulation_results, AX=AX, **kwargs)
    
    return figs, AX, simulation_results

###############################################################################
#.ini Parsing
###############################################################################
def read_ini(filename):
    '''
    Reads a .ini file and extracts model data relevant to simulation. Returns a 
    dictionary of keyword arguments that can be passed into downstream functions.

    Parameters
    ----------
    filename : str or Path-like
        The file to be read.

    Returns
    -------
    model_data : dict
        The data from the file read using model_handler.read_ini.
    sim_args : dict
        A dict in the form: 
        {<model_key>: {'model'     : <Model>, 
                       'objectives': <objectives>}
        }
        where <objectives> is a dict of <objective_name>: <objective_function> 
        pairs. 
    
    Notes
    -----
    Passes the model_data variable to get_sim_args so as to obtain sim_args
    
    See Also
    --------
    get_sim_args
    '''
    model_data = mh.read_ini(filename)
    sim_args   = get_sim_args(model_data)
    return model_data, sim_args

def get_sim_args(model_data):
    '''
    Reads data extracted from a file.Returns a dictionary of keyword arguments 
    that can be passed into downstream functions.

    Parameters
    ----------
    model_data : dict
        The data from the file.

    Returns
    -------
    sim_args : dict
        A dict in the form: 
        {<model_key>: {'model'     : <Model>, 
                       'objectives': <objectives>}
        }
        where <objectives> is a dict of <objective_name>: <objective_function> 
        pairs. 
    '''
    sim_args = {}
    for key, value in model_data.items():
        sim_args[key] = {'model'      : value['model'],
                         'objectives' : value.get('objectives')
                         }
        if 'args' in value:
            sim_args['args'] = value['args']
    return sim_args
    
###############################################################################
#Integration
###############################################################################
def integrate_models(sim_args):
    '''
    The main function for numerical integration.

    Parameters
    ----------
    sim_args : dict
        A dict in the form: 
        {<model_key>: {'model'     : <Model>, 
                       'objectives': <objectives>}
        }
        where <objectives> is a dict of <objective_name>: <objective_function> 
        pairs.
    
    Returns
    -------
    simulation_results : dict
        A dict nested as model_key -> scenario -> estimate -> (table, obj_vals) 
        where table is a DataFrame of the time response and obj_vals is a dict 
        containing the return values of the objective functions.
    '''
    simulation_results = {}
    for key, value in sim_args.items():
        
        check_model(value['model'])
        
        simulation_results[key] = integrate_model(**value)
    return simulation_results

def integrate_model(model, objectives=None, args=(), _tspan=None, _init=None, _params=None, _inputs=None):
    '''
    Handles numerical integration for an individual model.

    Parameters
    ----------
    model : Model
        A Model object to be integrated using the values stored in its tspan, 
        init_vals, param_vals and input_vals attributes.
    objectives : dict, optional
        A dict in the form {<objective_name>: <objective_func>}. The default is None.
    args : tuple, optional
        Additional arguments for integration. The default is ().
    _tspan : list of numpy.ndarray, optional
        Overrides the tspan in the Model object. For backend use. The default is None.
    _init : TYPE, optional
        Overrides the init_vals in the Model object. For backend use. The default is None.
    _params : TYPE, optional
        Overrides the param_vals in the Model object. For backend use. The default is None.
    _inputs : TYPE, optional
        Overrides the input_vals in the Model object. For backend use. The default is None.

    Returns
    -------
    simulation_result : dict
        A dict nested according to scenario -> estimate -> (table, obj_vals) where 
        table is a DataFrame of the time response and obj_vals is a dict containing 
        the return values of the objective functions.
    '''
    #Format and check
    init_vals  = model.init_vals[list(model.states)]
    param_vals = model.param_vals[list(model.params)]
    
    tspan   = model.tspan                                       if _tspan  is None else _tspan
    init_   = dict(zip(init_vals.index,  init_vals.values))     if _init   is None else _init
    params_ = dict(zip(param_vals.index, param_vals.values))    if _params is None else _params
    
    if model.inputs:
        input_vals = model.input_vals[list(model.inputs)]
        inputs_ = {i: df.values for i, df in input_vals.groupby(level=0)} if _inputs is None else _inputs
    else:
        inputs_ = None
        
    #Iterate across each scenario
    result = {}
    
    for scenario, init in init_.items():
        inputs           = inputs_[scenario] if model.inputs else None
        result[scenario] = {}
        
        for estimate, params_array in params_.items():
           
            y_model, t_model = itg.piecewise_integrate(model.func, 
                                                       tspan       = tspan, 
                                                       init        = init, 
                                                       params      = params_array, 
                                                       inputs      = inputs, 
                                                       scenario    = scenario,
                                                       modify      = model.modify,
                                                       args        = args,
                                                       overlap     = True,
                                                       **model.solver_args
                                                       )
            
            #Tabulate
            table             = tabulate(y_model, t_model, params_array, inputs, scenario, model, args)
            objective_results = evaluate_objective(table, objectives)
        
            result[scenario][estimate] = table, objective_results
        
    return result

###############################################################################
#Supporting Functions for Integration
###############################################################################
def evaluate_objective(table, objectives):
    '''
    :meta private:
    '''
    if not objectives:
        return None
    
    objective_results = {}
    for key, func in objectives.items():
        try:
            objective_results[key] = func(table)
        except Exception as e:
            msg    = 'Error in evaluating objective function "{}".'.format(key)
            args   = (msg,) + e.args
            e.args = ('\n'.join(args),)
            
            raise e
    return objective_results
    
def tabulate(y_model, t_model, params, inputs, scenario, model, args=()):
    '''
    :meta private:
    '''
    y_model_   = y_model.T
    tspan      = model.tspan
    p_array    = np.zeros((len(y_model_), len(params)))
    y_last     = y_model_[0]
    p_last     = params
    seg_start  = 0
    seg_stop   = len(tspan[0]) 
    
    if inputs is None:
        for segment in range(len(tspan)):
            _, y_args  = itg.int_args_helper(y_last, p_last, inputs, segment, scenario, modify=model.modify, args=args)#model.modify(y_last, p_last, i_last, scenario, segment)
            p_         = y_args[:1]
            seg_stop   = seg_start + len(tspan[segment])
            
            p_array[seg_start: seg_stop] = p_
            
            y_last    = y_model_[seg_stop-1]  
            p_last    = p_
            seg_start = seg_stop 
    
        table = np.concatenate((t_model[:,None], y_model_, p_array), axis=1)
        cols  = ('Time',) + model.states + model.params
    else:
        i_array    = np.zeros((len(y_model_), len(inputs[0])))
        
        for segment in range(len(tspan)):
            _, y_args  = itg.int_args_helper(y_last, p_last, inputs, segment, scenario, modify=model.modify, args=args)#model.modify(y_last, p_last, i_last, scenario, segment)
            p_, i_     = y_args[:2]
            seg_stop   = seg_start + len(tspan[segment])
            
            p_array[seg_start: seg_stop] = p_
            i_array[seg_start: seg_stop] = i_
            
            y_last    = y_model_[seg_stop-1]
            p_last    = p_
            seg_start = seg_stop 
    
        table = np.concatenate((t_model[:,None], y_model_, p_array, i_array), axis=1)
        cols  = ('Time',) + model.states + model.params + model.inputs  
    
    table = pd.DataFrame(table, columns=cols)
    
    return table

def check_model(model):
    '''
    :meta private:
    '''
    if model.inputs:
        if len(model.init_vals) != len(model.input_vals.index.unique(0)):
            msg = 'Number of scenarios do not match for model {}.'
            raise ValueError(msg.format(model.name))
    
###############################################################################
#Plotting
###############################################################################
def plot_simulation_results(plot_index, simulation_results, AX=None, **line_args):
    '''
    Plots the simulation results.    

    Parameters
    ----------
    plot_index : dict
        A dict in the form {<model_key>: <states>} where a state is a column name 
        in the appropriate table or a key name in the appropriate obj_vals. 
    simulation_results : dict
        A dict nested as model_key -> scenario -> estimate -> (table, obj_vals). 
        Where table is a DataFrame of the time response and obj_vals is a dict 
        containing the return values of the objective functions.
    AX : dict, optional
        A dict in the form {<model_key>: {<state>: <matplotlib Axes>}}. Tells the 
        function where to plost the results. If this argument is None, the Axes 
        objects are generated automatically.
        The default is None.
    **line_args : dict
        Keyword arguments for controlling the appearance of the 2D-line objects 
        to be generated. str and numerical values are applied to every line. If 
        you want to create different appeaarnces according to model and scenario, 
        use a dictionary for the value in the form {<model_key>: {<scenario>: value}}

    Returns
    -------
    figs : Figure
        Figure objects generated (if any).
    AX : dict
        A dict in the form {<model_key>: {<state>: <matplotlib Axes>}}.

    '''
    figs, AX1 = (None, AX) if AX else utp.make_AX(plot_index)
     
    for model_key, variables in plot_index.items():
        simulation_result = simulation_results[model_key]
        AX_               = utp.parse_recursive(AX1, model_key, apply=False)
        line_args_        = utp.parse_recursive(line_args, model_key)
        
        try:
            plot_simulation_result(variables, simulation_result, AX_, **line_args_)
        except Exception as e:
            msg    = 'Error in plotting {}'.format(model_key)
            args   = (msg,) + e.args
            
            e.args = ('\n'.join(args),)
            
            raise e
    
    # utp.apply_legend(AX)
    return figs, AX1

def plot_simulation_result(variables, simulation_result, AX, **line_args):
    '''
    Plots simulation result for a single model.

    Parameters
    ----------
    variables : list of str
        A list of states where each state is a column in table or a key in obj_vals.
    simulation_result : dict
        A dict nested according to scenario -> estimate -> (table, obj_vals) where 
        where table is a DataFrame of the time response and obj_vals is a dict 
        containing the return values of the objective functions.
    AX : dict
        A dict in the form {<state>: <matplotlib  Axes>}. Tells the function where 
        to plot the results.
    **line_args : dict
        Keyword arguments for controlling the appearance of the 2D-line objects 
        to be generated. str and numerical values are applied to every line. If 
        you want to create different appeaarnces according to model and scenario, 
        use a dictionary for the value in the form {<scenario>: value}
    
    Returns
    -------
    AX : dict
        A dict in the form {<state>: <matplotlib  Axes>}.
    '''
    label_scheme             = line_args.get('label')

    for scenario, scenario_result in simulation_result.items():
        first = True
        for estimate, estimate_result in scenario_result.items():
            for variable in variables:
                
                r  = get_result(variable, estimate_result, (scenario, estimate))
                ax = utp.parse_recursive(AX, variable, scenario, estimate, apply=False)

                #Parse labeling
                if not label_scheme:
                    label = {'label': '{} {} {}'.format(variable, scenario, estimate)}
                elif label_scheme == 'state':
                    label = {'label':'{}'.format(variable)} if first else {'label': '_nolabel'}
                elif label_scheme == 'state, scenario':
                    label = {'label':'{} {}'.format(variable, scenario)} #if first else {'label': '_nolabel'}
                elif label_scheme == 'state, estimate':
                    label = {'label':'{} {}'.format(variable, estimate)}
                elif label_scheme == 'scenario':
                    label = {'label':'{}'.format(scenario)} if first else {'label': '_nolabel'}
                elif label_scheme == 'scenario, estimate':
                    label = {'label': '{} {}'.format(scenario, estimate)}
                elif label_scheme == 'estimate':
                    label = {'label':'{}'.format(estimate)}
                else:
                    label = {}

                try:
                    if type(r) == dict:
                        args = {**line_args, **label, **r}
                        ax.plot(**args)
                    else:
                        
                        args = {**line_args, **label}
                        args = utp.parse_recursive(args, scenario, estimate, variable)
                        ax.plot(*r, **args)
                    ax.legend()
                except Exception as e:
                    msg    = ('Error in plotting scenario {}, variable {}'.format(scenario, variable), ) + e.args
                    e.args = ('\n'.join(msg) ,) 
                    raise e
            
            first = False
                
    return AX

###############################################################################
#Supporting Functions for Plotting
###############################################################################
def get_result(variable, estimate_result, error_key=''):
    '''
    :meta private:
    '''
    table, obj_vals = estimate_result
    obj_vals_       = obj_vals if obj_vals else []

    if variable in table:
        return table['Time'], table[variable]
    elif variable in obj_vals_:
        return obj_vals[variable]
    else:
        msg = 'Variable {} is not in {}.'
        raise Exception(msg.format(variable, error_key))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.close('all')
    
    #Preprocessing
    model_data = mh.read_ini('_test/TestModel_1.ini')
    model      = model_data['model_1']['model']
    
    def obj1(table):
        s = table['s']
        
        mu_max = table['mu_max']
        ks     = table['ks']
        
        t = table['Time']
        
        mu = mu_max*s/(s+ks)
        
        return t, mu
    
    def obj2(table):
        x = table['x']
        s = table['s']
        
        mu_max = table['mu_max']
        ks     = table['ks']
        ys     = table['ys']
        
        t = table['Time']
        
        mu = mu_max*s/(s+ks)
    
        dx = mu*x - 0.08*x
        
        return t, dx/ys
    
    def modify1(init, params, inputs, scenario, segment):
        new_init   = init.copy()
        new_params = params.copy()
        new_inputs = inputs.copy()
        
        new_init[0] *= 4
        
        return new_init, new_params, new_inputs
    
    # #Test integration
    # simulation_result = integrate_model(model)
    # scenario          = 0
    # estimate          = 0
    # table             = simulation_result[scenario][estimate][0]
    # assert table.shape == (62, 9)
    
    # #Test simulation with objective function
    # objectives        = {1 : obj1, 2: obj2}
    # simulation_result = integrate_model(model, objectives=objectives)
    # scenario          = 0
    # estimate          = 0
    # table, obj_vals   = simulation_result[scenario][estimate]
    
    # xo1, yo1 = obj_vals[1]
    # xo2, yo2 = obj_vals[2]
    
    # #Plot
    # fig = plt.figure()
    # AX  = [fig.add_subplot(5, 1, i+1) for i in range(5)]
    
    # AX[0].plot(table['Time'], table['x'])
    # AX[1].plot(table['Time'], table['s'])
    # AX[2].plot(table['Time'], table['b'])
    # AX[3].plot(xo1.values, yo1.values)
    # AX[4].plot(xo2.values, yo2.values)
    
    # #Test modifier
    # model.modify      = modify1
    # objectives        = {1 : obj1, 2: obj2}
    # simulation_result = integrate_model(model, objectives=objectives)
    # scenario          = 0
    # estimate          = 0 
    # table, obj_vals   = simulation_result[scenario][estimate]
    
    # xo1, yo1 = obj_vals[1]
    # xo2, yo2 = obj_vals[2]
    
    # assert xo1.shape == (62,)
    # assert yo1.shape == (62,)
    # assert xo2.shape == (62,)
    # assert yo2.shape == (62,)
    # model.modify = None
    
    # #Plot
    # fig = plt.figure()
    # AX  = [fig.add_subplot(5, 1, i+1) for i in range(5)]
    
    # AX[0].plot(table['Time'], table['x'])
    # AX[1].plot(table['Time'], table['s'])
    # AX[2].plot(table['Time'], table['b'])
    # AX[3].plot(xo1.values, yo1.values)
    # AX[4].plot(xo2.values, yo2.values)
    
    # #Test multi-model
    # sim_args = {'model_1'    : {'model': model,
    #                             'objectives' : {1 : obj1, 2: obj2}
    #                             },
    #             'model_2'    : {'model': model,
    #                             'objectives' : {1 : obj1, 2: obj2}
    #                             }
    #             }
    
    # simulation_results = integrate_models(sim_args)
    # model_key          = 'model_1'
    # scenario           = 0
    # estimate           = 0 
    # table, obj_vals    = simulation_results[model_key][scenario][estimate]
    
    # xo1, yo1 = obj_vals[1]
    # xo2, yo2 = obj_vals[2]
    
    # assert xo1.shape == (62,)
    # assert yo1.shape == (62,)
    # assert xo2.shape == (62,)
    # assert yo2.shape == (62,)
    
    #Test plotting
    sim_args = {'model_1'    : {'model': model,
                                'objectives' : {1 : obj1, 2: obj2}
                                },
                'model_2'    : {'model': model,
                                'objectives' : {1 : obj1, 2: obj2}
                                }
                }
    
    simulation_results = integrate_models(sim_args)
    
    #Test basic plot
    plot_index = {'model_1': ['x', 's', 'b', 1, 2]}
    figs, AX   = plot_simulation_results(plot_index, simulation_results)

    assert len(AX['model_1']['x'].lines) == 4
    assert len(AX['model_1']['b'].lines) == 4
    assert len(AX['model_1'][  2].lines) == 4
    
    # #Test line args
    # plot_index = {'model_1': ['x', 'b', 1, 2],
    #               'model_2': ['x', 'b', 1, 2]
    #               }
    # color      = {'model_1': {0: colors['cobalt'],
    #                           1: colors['marigold']
    #                           },
    #               'model_2': colors['teal']
    #               }
    # figs, AX   = plot_simulation_results(plot_index, simulation_results, color=color, label='scenario')
    
    # assert AX['model_1']['x'].lines[-1].get_label() == '_nolabel'
    
    # #Test high-level
    # model_data, sim_args = read_ini('_test/TestModel_2.ini')
    # plot_index           = {'model_1': ['x', 's', 'b', 'growth']}
    # simulation_results   = integrate_models(sim_args)
    # figs, AX             = plot_simulation_results(plot_index, simulation_results, color={'model_1': colors['cobalt']})
    
    