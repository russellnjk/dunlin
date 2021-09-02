import numpy             as     np
import pandas            as     pd
import scipy.optimize    as     sop
from numba               import njit
from scipy.stats         import norm
from time                import time, sleep
  
###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin.model                    as dml
import dunlin.simulate                 as sim
import dunlin._utils_optimize.wrap_SSE as ws
import dunlin._utils_optimize.algos    as ag 
import dunlin._utils_plot.plot         as upp
from dunlin._utils_optimize.params import SampledParam, Bounds, DunlinOptimizationError

###############################################################################
#Globals
###############################################################################
make_AX = upp.make_AX
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
def fit_model(model, dataset, n=1, algo='differential_evolution', AX=None, guess=':', **kwargs):
        
    opt_results = {}
    sse_calc    = ws.SSECalculator(model, dataset)
    opt_result  = OptResult.from_model(model, sse_calc) 
    
    for i in range(n):
        if n > 1:
            opt_result = opt_result.seed()
        if algo is not None:
            run_algo(opt_result, algo, **kwargs)
            
        opt_results[i] = opt_result
        
        if AX:
            args = {'label' : '_nolabel'} if i else {}
            integrate_and_plot(model, AX, opt_results={i: opt_result}, dataset=dataset, guess=guess, **args)
            sleep(0.1)
    return opt_results

def integrate_opt_result(model, opt_result, _tspan=None):
    best_params, _ = opt_result.get_best()
    sim_results    = sim.integrate_model(model, _params=best_params, _tspan=_tspan)
    
    return sim_results
    
def integrate_and_plot(model, AX, opt_results={}, dataset=None, guess=':', **line_args):
    line_args_      = {**line_args, **{'label': 'scenario'}}
    guess_line_args = {**model.sim_args['line_args'], **line_args_, **{'linestyle': guess}}
    data_line_args  = {**model.sim_args['line_args'], **line_args_, **{'marker': 'o', 'linestyle': 'None'}}
    
    #Determine appropriate tspan
    tspan      = {}
    opt_result = next(iter(opt_results.values())) if opt_results else None
    blank      = np.array([]) 
    for scenario in model.states.index:
        mtspan = model.get_tspan(scenario)
        if opt_result:
            otspan = getattr(opt_result.neg_log_likelihood, 'tspan', {}).get(scenario,  np.array([]))
        else:
            otspan = blank
            
        tspan[scenario] = max(mtspan, otspan, key=len)
        
    #Integrate opt results
    first = True
    if opt_results:
        for run, opt_result in opt_results.items():
            if not first:
                line_args_['label'] = '_nolabel'
                
            #Integrate and plot
            sim_results = integrate_opt_result(model, opt_result, tspan)
            sim.plot_sim_results(sim_results, AX, **line_args_)
            first = False
        
    #Integrate guess values
    if guess not in ['None', '']:
        if opt_results:
            guess_line_args['label'] = '_nolabel'
            
        sim_results = sim.integrate_model(model, _tspan=tspan)
        sim.plot_sim_results(sim_results, AX, **guess_line_args)
        
    #Overlay the data
    if dataset:
        if opt_results or guess not in ['None', '']:
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
    while Bayesian approaches attempt to MAXIMIZE the objective function.
    
    Therefore, the NEGATIVE of the log-likelihood should be used instead of the 
    regular log-likelihood for instantiation. For curve-fitting, this is 
    simply the SSE function.
    '''
    ###########################################################################
    #Instantiation
    ###########################################################################
    @classmethod
    def from_model(cls, model, to_minimize=None):
        #Check to_minimize
        if callable(to_minimize):
            to_minimize_ = to_minimize
        else:
            raise NotImplementedError('Still in the works.')
            
        #Instantiate
        nominal    = model.params
        kwargs     = {**getattr(model, 'optim_args', {}), **{'to_minimize': to_minimize_, 'nominal': nominal}}
        opt_result = cls(**kwargs)
        
        return opt_result
    
    def __init__(self, nominal, free_params, to_minimize, **kwargs):
        self.settings           = kwargs.get('settings', {})
        self.line_args          = kwargs.get('line_args', {})
        self.neg_log_likelihood = to_minimize
        self.result             = {}
        self.fixed              = []
        self.free_params        = {}
        self.sampled_params     = []
        
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
        return f'{type(self).__name__}<{lst}>'
    
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
    def seed(self):
        nominal = self.nominal.copy()
        for sp in self.sampled_params:
            new_val          = sp.new_sample()
            nominal[sp.name] = new_val
            
        args = {'free_params' : self.free_params, 'to_minimize' : self.neg_log_likelihood,
                'settings'    : self.settings,    'line_args'   : self.line_args
                }
        return type(self)(nominal, **args)        

###############################################################################
#Plotting
###############################################################################       
def plot_traces(opt_results, AX, palette=None, **line_args):
    global colors
    scale_types = {'lin' : 'linear', 'log10': 'log', 'log': 'log'}
    AX1         = AX
    for run, opt_result in opt_results.items(): 
        for var, ax_ in AX1.items():
            ax         = upp.recursive_get(ax_, run, var) 
            line_args_ = {**opt_result.line_args, **line_args}
            line_args_ = {k: upp.recursive_get(v, run, var) for k, v in line_args_.items()}
            
            #Process special keywords
            color = line_args_.get('color')
            if type(color) == str:
                line_args_['color'] = colors[color]
                
            label_scheme   = line_args_.get('label', 'scenario, run')
            if label_scheme == 'run':
                label = f'{run}'
            elif label_scheme == 'model_key':
                label = f'{opt_result.model_key}'
            elif label_scheme == 'model_key, run':
                label = f'{opt_result.model_key}, {run}'
            else:
                label = f'{run}'
            
            line_args_['label'] = label
            plot_type           = line_args_.get('plot_type', 'line')
            
            #Plot
            if plot_type == 'line':
                if line_args_.get('marker', None) and 'linestyle' not in line_args_:
                    line_args_['linestyle'] = 'None'
                
                if type(var) == tuple:
                    #Axis scale
                    scale = opt_result.free_params[var[0]].get('scale', 'lin')
                    scale = scale_types[scale]
                    ax.set_xscale(scale)
                    scale = opt_result.free_params[var[1]].get('scale', 'lin')
                    scale = scale_types[scale]
                    ax.set_yscale(scale)
                    
                    #Plot
                    x_vals, y_vals = opt_result[var[0]].values, opt_result[var[1]].values
                    ax.plot(x_vals, y_vals, **line_args_)
                    ax.set_xlabel(var[0])
                    ax.set_ylabel(var[0])
                else:
                    #Axis scale
                    scale = opt_result.free_params[var].get('scale', 'lin')
                    scale = scale_types[scale]
                    ax.set_yscale(scale)
                    
                    #Plot
                    y_vals = opt_result[var].values
                    ax.plot(y_vals, **line_args_)
                    ax.set_ylabel(var)
            else:
                raise ValueError(f'Unrecognized plot_type {plot_type}')
        
    return AX1    
    
def plot_dataset(dataset, AX, **line_args):
    global colors
    
    AX1 = AX
    
    for (dtype, scenario, var), data in dataset.items():
        if dtype != 'Data':
            continue
        
        ax_ = upp.recursive_get(AX1, var, scenario) 
        if type(ax_) == dict:
            ax_ = ax_.values()
        else:
            ax_ = [ax_]
                      
        line_args_  = {**getattr(dataset, 'line_args', {}), **line_args}
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
            
            for ax in ax_:
                ax.errorbar(x_vals, y_vals, y_err_, x_err_, **line_args_)
                # ax.set_ylabel(var)
        else:
            raise ValueError(f'Unrecognized plot_type {plot_type}')
        
    return AX1
    