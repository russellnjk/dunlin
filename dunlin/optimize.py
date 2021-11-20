import numpy             as     np
import pandas            as     pd
import scipy.optimize    as     sop
from numba               import njit
from scipy.stats         import norm
from time                import time
  
###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin.simulate                 as sim
import dunlin._utils_optimize.wrap_SSE as ws
import dunlin._utils_optimize.algos    as ag 
import dunlin._utils_plot              as upp
from dunlin._utils_optimize.params import SampledParam, Bounds, DunlinOptimizationError

###############################################################################
#Globals
###############################################################################
colors  = upp.colors

###############################################################################
#Wrapping
###############################################################################
def timer(func):
    def helper(*args, **kwargs):
        print(f'Starting {func.__name__}')
        start  = time()
        result = func(*args, **kwargs)
        stop   = time()
        delta  = '{:.2e}'.format(stop-start)
        print(f'Time taken for {func.__name__}', delta)
        return result
    return helper

def run_algo(opt_result, algo, **kwargs):
    func = getattr(opt_result, algo, None)
    if func is None:
        raise DunlinOptimizationError.no_algo(algo)
    func(**kwargs)
    
###############################################################################
#High-level Functions
###############################################################################        
def fit_model(model, dataset, n=1, algo='differential_evolution',guess=':', **kwargs):
        
    opt_results = []
    sse_calc    = ws.SSECalculator(model, dataset)
    opt_result  = OptResult.from_model(model, sse_calc, name=0) 
    
    for i in range(n):
        if i > 0:
            opt_result = opt_result.seed(i)
        
        run_algo(opt_result, algo, **kwargs)
        
        opt_results.append(opt_result)
        
    return opt_results

def simulate_and_plot(model, AX, optresults=None, dataset=None, guess_marker=':', **sim_args):
    guess_line_args = {**sim_args.get('line_args', {}), **{'linestyle': guess_marker}}
    data_line_args  = {**model.sim_args.get('line_args', {}), **sim_args.get('line_args', {}), **{'marker': 'o', 'linestyle': 'None'}}
    
    #Determine appropriate tspan
    tspan      = {}
    opt_result = optresults[0] if optresults else None
    blank      = np.array([]) 
    for scenario in model.states.index:
        mtspan = model.get_tspan(scenario)
        if opt_result:
            otspan = getattr(opt_result.neg_log_likelihood, 'tspan', {}).get(scenario,  np.array([]))
        else:
            otspan = blank
            
        tspan[scenario] = max(mtspan, otspan, key=len)
        
    #Integrate opt results
    first     = True
    sim_args_ = sim_args.copy()
    if optresults:
        for optr in optresults:
            if not first:
                sim_args_['line_args']['label'] = '_nolabel'
            
            #Integrate and plot
            srs = optr.simulate(model)
            sim.plot_simresults(srs, AX, **sim_args_)
            first = False
        
    #Integrate guess values
    if optresults:
        guess_line_args['label'] = '_nolabel'
        
        sim_results = sim.simulate_model(model)
        sim.plot_simresults(sim_results, AX, **{'line_args': guess_line_args})
        
    #Overlay the data
    if dataset:
        if optresults or guess_marker not in ['None', '']:
            data_line_args['label'] = '_nolabel'
        plot_dataset(dataset, AX, **data_line_args)            

def get_best(opt_results):
    if type(opt_results) == OptResult:
        return get_best({0: opt_results})
    
    best_params, best_posterior = {}, {}
    for run, opt_result in opt_results.items():
        params, posterior   = opt_result.get_best()
        best_params[run]    = params
        best_posterior[run] = posterior
        
    return best_params, best_posterior
    
###############################################################################
#Dunlin Classes
###############################################################################    
class OptResult():
    '''
    Note: Many optimization algorithms seek to MINIMIZE an objective function, 
    while Bayesian formula attempts to MAXIMIZE the objective function.
    
    Therefore, the NEGATIVE of the log-likelihood should be used instead of the 
    regular log-likelihood for instantiation. For curve-fitting, this is 
    simply the SSE function.
    '''
    scale_types = {'lin' : 'linear', 'log10': 'log10', 'log': 'log'}
    
    ###########################################################################
    #Instantiation
    ###########################################################################
    @classmethod
    def from_model(cls, model, to_minimize, name=0):
        #Check to_minimize
        if not callable(to_minimize):
            raise ValueError('to_minimize must be callable.')
        
        #Instantiate
        nominal     = model.params
        free_params = model.optim_args.get('free_params', {})
        settings    = model.optim_args.get('settings',    {})
        trace_args  = model.optim_args.get('trace_args',  {})
        name        = name
        opt_result = cls(nominal, free_params, to_minimize, name, settings, trace_args)
        
        return opt_result
    
    def __init__(self, nominal, free_params, to_minimize, name=0, settings=None, trace_args=None):
        self.settings           = {} if settings   is None else settings
        self.trace_args         = {} if trace_args is None else trace_args
        self.neg_log_likelihood = to_minimize
        self.result             = {}
        self.fixed              = []
        self.free_params        = {}
        self.sampled_params     = []
        self.name               = name 
        
        if not free_params:
            raise ValueError('No free parameters provided.')
            
        for p in nominal.keys():
            if p in free_params:
                kw = free_params[p]
                self.sampled_params.append(SampledParam(p, **kw))
                self.free_params[p] = free_params[p]
            else:
                self.fixed.append(p)
        
        try:
            self.nominal = pd.DataFrame(nominal)
        except:
            try:
                self.nominal = pd.DataFrame([nominal])
            except:
                raise DunlinOptimizationError.nominal()
    
    ###########################################################################
    #Optimization and Calculation
    ###########################################################################       
    def get_objective(self, free_params_array, _a=None, _p=None):
        priors   = np.zeros(len(free_params_array))
        unscaled = np.zeros(len(free_params_array))
        
        for i, (x, sp) in enumerate( zip(free_params_array, self.sampled_params) ):
            priors[i]   = sp(x)
            unscaled[i] = sp.unscale(x)
        
        neg_posterior = self.neg_log_likelihood(unscaled) - self.logsum(priors)
        
        if _a is not None:
            _a.append(free_params_array)
        if _p is not None:
            _p.append(neg_posterior)
        
        return neg_posterior
    
    @staticmethod
    @njit
    def logsum(arr):
        return np.sum(np.log(arr))
    
    def get_bounds(self):
        return [p.get_opt_bounds() for p in self.sampled_params]
    
    def get_x0step(self):
        x0   = []
        step = []
        for p in self.sampled_params:
            b = p.get_opt_bounds()
            step.append( (b[1] - b[0])/40 )
            x0.append(sum(b)/2)
    
        return np.array(x0), np.array(step)
    
    def format_result(self, a, p, **kwargs):
        #Unscale
        a, p = np.array(a), np.array(p)
        
        for i, sp in enumerate(self.sampled_params):
            a[:,i] = sp.unscale(a[:,i])
        
        #Cache and return the result
        result = {'a': pd.DataFrame(a, columns=self.names),
                  'p': p
                  }
        result = {**result, **kwargs}
        self.result = result
        return result
    
    ###########################################################################
    #Wrapped Algorithms
    ###########################################################################     
    @timer
    def differential_evolution(self, **kwargs):
        a        = []
        p        = []
        func     = lambda x: self.get_objective(x, a, p)
        bounds   = self.get_bounds()
        settings = {**{'bounds': bounds}, **self.settings, **kwargs}
        result   = sop.differential_evolution(func, **settings)

        #Cache and return the result
        return self.format_result(a, p, o=result)
    
    @timer
    def basinhopping(self, **kwargs):
        a        = []
        p        = []
        func     = lambda x: self.get_objective(x, a, p)
        bounds   = Bounds(self.sampled_params)
        x0, step = self.get_x0step()
        gen      = norm(0, step)
        step     = lambda x: x + gen.rvs()
        settings = {**{'accept_test'      : bounds, 
                       'minimizer_kwargs' : {}, 
                       'x0'               : x0, 
                       'take_step'        : step
                       }, 
                    **self.settings, 
                    **kwargs
                    }
        settings['minimizer_kwargs'].setdefault('bounds', bounds)
        
        #Check initial guess
        if type(settings['x0']) == dict:
            settings['x0'] = [settings['x0'][k] for k in self.nominal.keys()]
        settings['x0'] = np.array(settings['x0'])
        
        if not bounds(x_new=settings['x0']):
            raise ValueError('Initial guess is out of bounds.')
        
        #Run algorithm
        result = sop.basinhopping(func, **settings)
        
        #Cache and return the result
        return self.format_result(a, p, o=result)
    
    @timer
    def dual_annealing(self, **kwargs):
        a        = []
        p        = []
        func     = lambda x: self.get_objective(x, a, p)
        bounds   = Bounds(self.sampled_params)
        x0, step = self.get_x0step()
        gen      = norm(0, step)
        step     = lambda x: x + gen.rvs()
        settings = {**{'bounds'               : bounds, 
                       'local_search_options' : {}, 
                       'x0'                   : x0
                       }, 
                    **self.settings, 
                    **kwargs
                    }
        
        #Check initial guess
        if type(settings['x0']) == dict:
            settings['x0'] = [settings['x0'][k] for k in self.nominal.keys()]
        settings['x0'] = np.array(settings['x0'])
        
        if not bounds(x_new=settings['x0']):
            raise ValueError('Initial guess is out of bounds.')
        
        #Run algorithm
        result = sop.basinhopping(func, **settings)
        
        #Cache and return the result
        return self.format_result(a, p, o=result)
        
    @timer
    def local_minimize(self, **kwargs):
        a        = []
        p        = []
        func     = lambda x: self.get_objective(x, a, p)
        bounds   = Bounds(self.sampled_params)
        x0, step = self.get_x0step()
        gen      = norm(0, step)
        step     = lambda x: x + gen.rvs()
        settings = {**{'bounds'      : bounds, 
                       'x0'          : x0
                       }, 
                    **self.settings, 
                    **kwargs
                    }
        
        #Check initial guess
        if type(settings['x0']) == dict:
            settings['x0'] = [settings['x0'][k] for k in self.nominal.keys()]
        settings['x0'] = np.array(settings['x0'])
        
        if not bounds(x_new=settings['x0']):
            raise ValueError('Initial guess is out of bounds.')
        
        #Run algorithm
        result = sop.basinhopping(func, **settings)
        
        #Cache and return the result
        return self.format_result(a, p, o=result)
    
    @timer
    def simulated_annealing(self, **kwargs):
        func     = self.get_objective 
        bounds   = Bounds(self.sampled_params)
        x0, step = self.get_x0step()
        gen      = norm(0, step)
        step     = lambda x: x + gen.rvs()
        settings = {**{'bounds' : bounds, 
                       'x0'     : x0, 
                       'step'   : step
                       }, 
                    **self.settings, 
                    **kwargs
                    }
        
        #Check initial guess
        if type(settings['x0']) == dict:
            settings['x0'] = [settings['x0'][k] for k in self.nominal.keys()]
        settings['x0'] = np.array(settings['x0'])
        
        if not bounds(x_new=settings['x0']):
            raise ValueError('Initial guess is out of bounds.')
            
        #Run algorithm
        a, p   = ag.simulated_annealing(func, **settings)
        
        #Cache and return the result
        return self.format_result(a, p)
    
    ###########################################################################
    #Access
    ###########################################################################      
    def __getitem__(self, key):
        try:
            df = self.result['a']
        except:
            raise DunlinOptimizationError.no_opt_result()
        
        return df[key]
    
    def __getattr__(self, attr):
        if attr == 'posterior':
            return self.result.get('p', None)
        elif attr == 'names':
            return [sp.name for sp in self.sampled_params]
        elif attr in self.result:
            return self.result[attr]
        else:
            raise AttributeError(f'"{type(self).__name__}" does not have attribute "{attr}"')
    
    ###########################################################################
    #Printing
    ###########################################################################    
    def __repr__(self):
        lst = ', '.join([sp.name for sp in self.sampled_params])
        return f'{type(self).__name__} {self.name}({lst})'
    
    def __str__(self):
        return self.__repr__()
    
    ###########################################################################
    #Result Extraction
    ###########################################################################      
    def get_best(self, with_nominal=True):
        best_params, best_posterior = self.get_nbest(with_nominal=with_nominal)
        best_posterior              = best_posterior[0]
        
        if with_nominal:
            idx            = best_params.index.levels[0][0]
            best_params    = best_params.loc[idx]
        else:
            best_params    = best_params.iloc[0]
        return best_params, best_posterior
        
    def get_nbest(self, n=1, k=1, with_nominal=True):
        idx            = np.argsort(self.posterior)[0:n:k]
        best_params    = self.a.loc[idx]
        best_posterior = self.posterior[idx]
        
        if with_nominal:
            free     = list(self.free_params)
            template = self.nominal[self.fixed]
            cols     = self.nominal.columns
            result   = {}
            for i, row in best_params.iterrows():
                to_merge              = template.copy()
                shape                 = to_merge.shape[0], len(row)
                vals                  = np.resize(row.values, shape)
                to_merge.loc[:, free] = vals
                result[i]             = to_merge[cols]
            
            best_params = pd.concat(result, axis=0)
        return best_params, best_posterior
    
    ###############################################################################
    #Seeding
    ###############################################################################       
    def seed(self, new_name=0):
        nominal = self.nominal.copy()
        for sp in self.sampled_params:
            new_val          = sp.new_sample()
            nominal[sp.name] = new_val
            
        args = {'free_params' : self.free_params, 'to_minimize' : self.neg_log_likelihood,
                'settings'    : self.settings,    'trace_args'  : self.trace_args,
                'name'        : new_name
                }
        return type(self)(nominal, **args)        
    
    ###############################################################################
    #Plotting
    ###############################################################################       
    def simulate(self, model):
        best_params, _ = self.get_best()
        p              = dict(zip(best_params.index, best_params.values))     
        return sim.simulate_model(model, p=p)
         
    def simulate_and_plot(self, model, AX, **sim_args):
        srs = self.simulate(model)
        
        return sim.plot_simresults(srs, AX, **sim_args)
    
    def plot_trace(self, var, ax, plot_type='line', **trace_args):
        if plot_type == 'line':
            if type(var) in [tuple, list]:
                if len(var) == 1:
                    return self.plot_trace_line(ax, None, var[0], **trace_args)
                    
                elif len(var) == 2:
                    return self.plot_trace_line(ax, *var, **trace_args)
                else:
                    raise NotImplementedError(f'Cannot plot {var}. Length must be one or two.')
                    
            elif type(var) == str:
                return self.plot_trace_line(ax, None, var, **trace_args)

            else:
                raise NotImplementedError(f'No implementation for {var}')
        else:
            raise ValueError(f'Unrecognized plot_type {plot_type}')
        
    def plot_trace_line(self, ax, x, y, **trace_args):
        scale_types = self.scale_types
        trace_args  = self._recursive_get(y, trace_args) if x is None else self._recursive_get((x, y), trace_args) 
        
        trace_args.setdefault('marker', '+')
        trace_args.setdefault('linestyle','None')
        
        if x is not None:
            scale = self.free_params[x].get('scale', 'lin')
            scale = scale_types[scale]
            ax.set_xscale(scale)
            ax.set_xlabel(x)
            x_vals = self[x].values
        
        scale = self.free_params[y].get('scale', 'lin')
        scale = scale_types[scale]
        ax.set_yscale(scale)
        y_vals = self[y].values
        ax.set_ylabel(y)
        
        #Plot
        if x is not None:
            return ax.plot(x_vals, y_vals, **trace_args)
        else:
            return ax.plot(y_vals, **trace_args)
    
    def _recursive_get(self, var, trace_args):
        global colors

        trace_args_ = {**self.trace_args, **trace_args}
        trace_args_ = {k: upp.recursive_get(v, self.name, var) for k, v in trace_args_.items()}
        
        #Process special keywords
        color = trace_args_.get('color')
        if type(color) == str:
            trace_args_['color'] = colors[color]
            
        label_scheme   = trace_args_.get('label', 'run')
        if label_scheme == 'run':
            label = f'Run {self.name}'
        elif label_scheme == 'model_key':
            label = f'Model {self.model_key}'
        elif label_scheme == 'model_key, run':
            label = f'Model {self.model_key}, Run {self.name}'
        else:
            label = f'Run {self.name}'
        
        trace_args_['label'] = label
        
        return trace_args_

###############################################################################
#Plotting
###############################################################################
def plot_traces(optresults, AX, plot_type='line', **trace_args):
    result = {}
    for var, ax_ in AX.items():
        for optr in optresults:
        
            ax   = upp.recursive_get(ax_, var, optr.name) 
            plot = optr.plot_trace(var, ax, plot_type=plot_type, **trace_args)
            
            result.setdefault(var, {})[optr.name] = plot
    return result
          
def plot_dataset(dataset, AX, **data_args):
    global colors
    
    plots = {}
    
    for (dtype, scenario, var), data in dataset.items():
        if dtype != 'Data':
            continue
        
        ax = upp.recursive_get(AX, var, scenario) 

        if not ax:
            continue
        
        line_args_  = {**getattr(dataset, 'line_args', {}), **data_args}
        line_args_  = {k: upp.recursive_get(v, scenario, var) for k, v in line_args_.items()}
        
        #Process special keywords
        color = line_args_.get('color')
        if type(color) == str:
            line_args_['color'] = colors[color]
        
        plot_type   = line_args_.get('plot_type', 'errorbar')
            
        #Plot
        if plot_type == 'errorbar':
            if line_args_.get('marker', None) and 'linestyle' not in line_args_:
                line_args_['linestyle'] = 'None'
            
            x_vals = dataset[('Time', scenario, var)]
            y_vals = data
            y_err_ = dataset.get(('Yerr', scenario, var))
            x_err_ = dataset.get(('Xerr', scenario, var))
            
            plots.setdefault(var, {})[scenario] = ax.errorbar(x_vals, y_vals, y_err_, x_err_, **line_args_)
        else:
            raise ValueError(f'Unrecognized plot_type {plot_type}')
        
    return plots
    