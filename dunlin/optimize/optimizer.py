import numpy             as     np
import pandas            as     pd
from collections         import namedtuple
from numba               import njit
from numbers             import Number
from scipy.stats         import norm
from time                import time
from typing              import Callable
 
###############################################################################
#Non-Standard Imports
###############################################################################
from .            import algos    as ag 
from .            import trace    as tr
from .params      import SampledParam, Bounds
from .sensitivity import SensitivityMixin

###############################################################################
#Typing
###############################################################################
Parameter = str
State     = str

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
    
###############################################################################
#High-level Functions
###############################################################################  
best_result = namedtuple('Best', 'parameters objective posterior n')      
def get_best_optimization(optimizers):
    best_parameters, best_objective, best_posterior, n = None, None, None, 0
    
    for i, optimizer in enumerate(optimizers):
        parameters, objective, posterior = optimizer.get_best()
        
        if best_parameters is None:
            best_parameters, best_objective, best_posterior, n = parameters, objective, posterior, i
        elif posterior > best_posterior:
            best_parameters, best_objective, best_posterior, n = parameters, objective, posterior, i 
    
    return best_result(best_parameters, best_objective, best_posterior, n)
    
###############################################################################
#Dunlin Classes
###############################################################################    
class Optimizer(SensitivityMixin):
    '''
    Note: Many optimization algorithms seek to MINIMIZE an objective function, 
    while Bayesian formula attempts to MAXIMIZE the posterior function.
    
    Therefore, the NEGATIVE of the log-likelihood should be used instead of the 
    usual log-likelihood for instantiation. For curve-fitting, this is 
    simply the SSE function.
    '''
    
    ###########################################################################
    #Instantiation
    ###########################################################################
    @classmethod
    def from_model(cls, model, to_minimize):
        #Check to_minimize
        if not callable(to_minimize):
            raise ValueError('to_minimize must be callable.')
        
        #Instantiate
        nominal         = model.parameter_df
        free_parameters = model.optim_args.get('free_parameters', {})
        settings        = model.optim_args.get('settings',    {})
        trace_args      = model.trace_args
        opt_result      = cls(nominal, free_parameters, to_minimize, settings, trace_args)
        
        return opt_result
    
    def __init__(self, 
                 nominal         : pd.DataFrame, 
                 free_parameters : list[Parameter], 
                 to_minimize     : Callable, 
                 settings        : dict = None, 
                 trace_args      : dict = None, 
                 ref             : str  = None
                 ):
        
        self.ref                 = ref
        self.settings            = {} if settings   is None else settings
        self.trace_args          = {} if trace_args is None else trace_args
        self.neg_log_likelihood  = to_minimize
        self.fixed               = []
        self.free_parameters     = {}
        self.sampled_parameters  = []
        sampled_parameter_idxs   = []
        
        #Check input and update attributes
        if not free_parameters:
            raise ValueError('No free parameters provided.')
            
        for i, p in enumerate(nominal.keys()):
            if p in free_parameters:
                kw = free_parameters[p]
                sp = SampledParam(p, **kw)
                self.sampled_parameters.append(sp)
                sampled_parameter_idxs.append(i)
                self.free_parameters[p] = free_parameters[p]
            else:
                self.fixed.append(p)
        
        if type(nominal) != pd.DataFrame:
            msg  = f'Error instantiating {type(self).__name__} for model {ref}. '
            msg += 'The "nominal" argument must be a DataFrame. Received {type(nominal)}.'
            raise TypeError(msg)
        
        self.nominal                = nominal.copy()
        self.nominal_dct            = dict(zip(nominal.index, nominal.values))
        self.sampled_parameter_idxs = np.array(sampled_parameter_idxs)

    ###########################################################################
    #Optimization and Calculation
    ###########################################################################       
    def __call__(self, free_parameters_array: np.ndarray) -> Number:
        return self.get_objective(free_parameters_array)
    
    def get_objective(self, free_parameters_array: np.ndarray) -> Number:
        priors   = np.zeros(len(free_parameters_array))
        unscaled = np.zeros(len(free_parameters_array))
        
        for i, (x, sp) in enumerate( zip(free_parameters_array, self.sampled_parameters) ):
            priors[i]   = sp(x)
            unscaled[i] = sp.unscale(x)
        
        neg_posterior = self.neg_log_likelihood(unscaled) - self.logsum(priors)
        
        return neg_posterior
    
    @staticmethod
    @njit
    def logsum(arr):
        return np.sum(np.log(arr))
    
    def get_bounds(self):
        return [p.scaled_bounds for p in self.sampled_parameters]
    
    def get_x0step(self, x0_nominal=False):
        x0   = []
        step = []
        for p in self.sampled_parameters:
            b = p.scaled_bounds()
            step.append( (b[1] - b[0])/40 )
            
            if x0_nominal:
                nominal = self.nominal[p.name]
                nominal = np.mean(nominal)
                nominal = p.scale(nominal)
                x0.append(nominal)
            else:
                x0.append(sum(b)/2)
    
        return np.array(x0), np.array(step)
    
    def _check_x0(self, x0, bounds):
        #Check initial guess
        if len(x0) != len(self.sampled_parameters):
            raise ValueError('Initial guess is not the same length as sampled parameters.')
        
        if not bounds(x_new=x0):
            over, under = bounds.get_out_of_bounds(x_new=x0)
            over_  = [f'{k} : Max {v[0]}, Received {v[1]}' for k, v in over.items() ]
            under_ = [f'{k} : Min {v[0]}, Received {v[1]}' for k, v in under.items()]
            
            over_  = '\n'.join(over_)
            under_ = '\n'.join(under_)
            msg    = f'Initial guess is out of bounds.\n{over_}\n{under_}'
            
            raise ValueError(msg)
    
    def _x02array(self, x0):
        if type(x0) == dict:
            x0 = np.array([x0[k] for k in self.nominal.keys()])
            
        elif type(x0) in [list, tuple]:
            x0 = np.array(x0)
        
        return x0
    
    ###########################################################################
    #Wrapped Algorithms
    ###########################################################################     
    @timer
    def run_differential_evolution(self, **kwargs):
        func     = lambda x: self.get_objective(x)
        bounds   = self.get_bounds()
        settings = {**{'bounds': bounds}, 
                    **self.settings.get('differential_evolution', {}), 
                    **kwargs
                    }
        result   = ag.differential_evolution(func, **settings)
        
        #Cache and return the result
        return self.make_trace(result)
    
    @timer
    def run_basinhopping(self, x0_nominal=True, **kwargs):
        func             = lambda x: self.get_objective(x)
        bounds           = Bounds(self.sampled_parameters)
        x0, step         = self.get_x0step(x0_nominal)
        gen              = norm(0, step)
        step             = lambda x: x + gen.rvs()
        minimizer_bounds = [tuple(pair) for pair in zip(bounds.xmin, bounds.xmax)]
        
        settings = {**{'take_step'        : step,
                       'minimizer_kwargs' : {}
                       }, 
                    **self.settings.get('basinhopping', {}), 
                    **kwargs
                    }
        settings['minimizer_kwargs'].setdefault('bounds', minimizer_bounds)
        
        #Convert dict to array if user provides x0 as a dict
        settings['x0'] = self._x02array(settings['x0'])
        
        #Check initial guess
        self._check_x0(settings['x0'], bounds)
        
        #Run algorithm
        result = ag.basinhopping(func, bounds, x0, **settings)
        
        #Cache and return the result
        return self.make_trace(result)
    
    @timer
    def run_dual_annealing(self, x0_nominal=True, **kwargs):
        func             = lambda x: self.get_objective(x)
        bounds           = Bounds(self.sampled_parameters)
        x0, step         = self.get_x0step(x0_nominal)
        minimizer_bounds = [tuple(pair) for pair in zip(bounds.xmin, bounds.xmax)]
        
        settings = {**{'minimizer_kwargs' : {}, 
                       }, 
                    **self.settings.get('dual_annealing', {}),
                    **kwargs
                    }
        settings['minimizer_kwargs'].setdefault('bounds', minimizer_bounds)
        
        #Convert dict to array if user provides x0 as a dict
        settings['x0'] = self._x02array(settings['x0'])
        
        #Check initial guess
        self._check_x0(settings['x0'], bounds)
        
        #Run algorithm
        result = ag.dual_annealing(func, bounds, x0, **settings)
        
        #Cache and return the result
        return self.make_trace(result)
        
    @timer
    def run_local_minimize(self, x0_nominal=True, **kwargs):
        func     = lambda x: self.get_objective(x)
        bounds   = Bounds(self.sampled_parameters)
        x0, step = self.get_x0step(x0_nominal)
        settings = {**{'bounds'      : bounds, 
                       'x0'          : x0
                       }, 
                    **self.settings.get('local_minimize', {}), 
                    **kwargs
                    }
        
        #Convert dict to array if user provides x0 as a dict
        settings['x0'] = self._x02array(settings['x0'])
        
        #Check initial guess
        self._check_x0(settings['x0'], bounds)
        
        #Run algorithm
        result = ag.local_minimize(func, **settings)
        
        #Cache and return the result
        return self.make_trace(result)
    
    @timer
    def run_simulated_annealing(self, x0_nominal=True, **kwargs):
        func     = self.get_objective 
        bounds   = Bounds(self.sampled_parameters)
        x0, step = self.get_x0step(x0_nominal)
        gen      = norm(0, step)
        step     = lambda x: x + gen.rvs()
        settings = {**{'bounds' : bounds, 
                       'x0'     : x0, 
                       'step'   : step
                       }, 
                    **self.settings.get('simulated_annealing', {}), 
                    **kwargs
                    }
        
        #Convert dict to array if user provides x0 as a dict
        settings['x0'] = self._x02array(settings['x0'])
        
        #Check initial guess
        self._check_x0(settings['x0'], bounds)
            
        #Run algorithm
        result = ag.simulated_annealing(func, **settings)
        
        #Cache and return the result
        return self.make_trace(result)
    
    ###########################################################################
    #Trace
    ###########################################################################     
    def make_trace(self, result: tuple):
        
        samples, objective, context, raw = result
        
        #Unscale
        samples = np.array(samples)
        
        for i, sp in enumerate(self.sampled_parameters):
            samples[:,i] = sp.unscale(samples[:,i])
         
        self._trace = tr.Trace(self.free_parameters, 
                               samples, 
                               objective, 
                               context, 
                               raw, 
                               self.ref,
                               **self.trace_args
                               )
        return self.trace
    
    ###########################################################################
    #Access
    ###########################################################################      
    @property
    def trace(self):
        if self._trace is None:
            msg = 'Cannot return trace before running optimization.'
            raise AttributeError(msg)
        return self._trace
        
    ###########################################################################
    #Printing
    ###########################################################################    
    def __repr__(self):
        lst = ', '.join([sp.name for sp in self.sampled_parameters])
        return f'{type(self).__name__}({lst})'
    
    def __str__(self):
        return self.__repr__()
    
    ###############################################################################
    #Seeding
    ###############################################################################       
    def seed(self, new_name=0):
        nominal = self.nominal.copy()
        for sp in self.sampled_parameters:
            new_val          = sp.new_sample()
            nominal[sp.name] = new_val
        
        args = {'free_parameters' : self.free_parameters, 
                'to_minimize'     : self.neg_log_likelihood,
                'settings'        : self.settings,    
                'trace_args'      : self.trace_args,
                'name'            : new_name
                }
        return type(self)(nominal, **args)        