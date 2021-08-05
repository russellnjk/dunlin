import numpy             as     np
import pandas            as     pd
import scipy.optimize    as     sop
from numba               import njit
from scipy.stats         import norm, laplace, lognorm, loglaplace, uniform

###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin.model                    as dml
import dunlin.simulation               as sim
import dunlin._utils_optimize.wrap_SSE as ws
import dunlin.optimize                 as opt
import dunlin._utils_model.base_error  as dbe
import dunlin._utils_plot.plot         as upp

###############################################################################
#Globals
###############################################################################
figure        = upp.figure
gridspec      = upp.gridspec
colors        = upp.colors
palette_types = upp.palette_types
fs            = upp.fs
make_AX       = upp.make_AX
scilimit      = upp.scilimit
save_figs     = upp.save_figs
truncate_axis = upp.truncate_axis

###############################################################################
#Dunlin Errors
###############################################################################
class DunlinOptimizationError(dbe.DunlinBaseError):
    
    @classmethod
    def prior_type(cls, arg, value, correct):
        return cls.raise_template(f'Invalid {arg}: {value}\nValue must be in {correct} ', 0)
    
    @classmethod
    def prior_format(cls, arg, value, correct):
        return cls.raise_template(f'Invalid {arg} format: {value}\nValue must be {correct} ', 1)
    
    @classmethod
    def no_opt_result(cls):
        return cls.raise_template('No optimization yet. Make sure you have run one of the optimization algorithms.', 10)
    
    @classmethod
    def no_algo(cls, algo):
        return cls.raise_template(f'No algorithm called "{algo}".', 11)
    
    
###############################################################################
#High-level Functions
###############################################################################    
def optimize_models(models, to_minimize=None, algo='differential_evolution'):
    all_opt_results = {}
    
    for model_key, model in models.items():
        #Parse arguments
        to_minimize_ = to_minimize[model_key] if hasattr(to_minimize, 'items') else to_minimize
        algo_        = algo[model_key]        if hasattr(algo,        'items') else algo
        
        #Call and assign
        all_opt_results[model_key] = optimize_model(model, to_minimize_, algo_)
        
    return all_opt_results

def fit_models(models, all_datasets, algo='differential_evolution'):
    all_opt_results = {}
    
    for model_key, model in models.items():
        #Parse arguments
        dataset = all_datasets[model_key]
        algo_   = algo[model_key] if hasattr(algo, 'items') else algo
        
        #Call and assign
        all_opt_results[model_key] = fit_model(model, dataset, algo_)
        
    return all_opt_results
    
def fit_model(model, dataset, algo='differential_evolution'):
    get_SSE     = ws.SSECalculator(model, dataset)
    opt_results = optimize_model(model, get_SSE, algo)
    return opt_results
        
def optimize_model(model, to_minimize=None, algo='differential_evolution'):
    opt_results = OptResult.from_model(model, to_minimize)
    
    for estimate, optimizer in opt_results.items():
        func = getattr(optimizer, algo, None)
        if func is None:
            raise DunlinOptimizationError.no_algo(algo)
        func()
    return opt_results

def integrate_opt_result(model, opt_result, n=10, k=2):
    best_params, best_posterior = opt_result.get_best(n)
    
    params      = best_params.iloc[::k]
    tspan       = opt_result.neg_log_likelihood.tspan
    sim_results = sim.integrate_model(model, _params=params, _tspan=tspan)
    
    return sim_results
    
###############################################################################
#Dunlin Classes
###############################################################################    
class SampledParam():
    
    _scale = {'lin'   : lambda x: x,
              'log'   : lambda x: np.log(x),
              'log10' : lambda x: np.log10(x)
              }
    
    _unscale = {'lin'   : lambda x: x,
                'log'   : lambda x: np.exp(x),
                'log10' : lambda x: 10**x
                }
    
    _priors = {'uniform'    : lambda lb, ub    : uniform(lb, ub-lb),
               'normal'     : lambda mean, sd  : norm(mean, sd),
               'laplace'    : lambda loc, scale: laplace(loc, scale),
               'logNormal'  : lambda mean, sd  : lognorm(mean, sd),
               'logLaplace' : lambda loc, scale: loglaplace(loc, scale),
               }
    
    _priors['parameterScaleUniform'] = _priors['uniform']
    _priors['parameterScaleNormal' ] = _priors['normal' ]
    _priors['parameterScaleLaplace'] = _priors['laplace' ]
    
    _priors['uni']     = _priors['uniform']
    _priors['norm']    = _priors['normal']
    _priors['lap']     = _priors['laplace']
    _priors['lognorm'] = _priors['logNormal']
    _priors['loglap']  = _priors['logLaplace']
    _priors['psuni']     = _priors['uniform']
    _priors['psnorm']    = _priors['normal']
    _priors['pslap']     = _priors['laplace']
    
    _priors['u']  = _priors['uniform']
    _priors['n']  = _priors['normal']
    _priors['l']  = _priors['laplace']
    _priors['ln'] = _priors['logNormal']
    _priors['ll'] = _priors['logLaplace']
    
    @classmethod
    def read_prior(cls, prior, scale, _name='prior'):
        try:
            if hasattr(prior, 'items'):
                ptype, a, b = prior['type'], prior['loc'], prior['scale']
            else:
                ptype, a, b = prior

            func = cls._priors[ptype]
        except KeyError:
            raise DunlinOptimizationError.prior_type(_name, prior, list(cls._priors.keys()))
        except:
            raise DunlinOptimizationError.prior_format(_name, prior, 'list/tuple in the order [type, loc, scale] or dict with keys type, loc and scale')
        return ptype, func(a, b)
            
    def __init__(self, name, bounds, prior=None, sample=None, scale='lin'):
        #Set name
        self.name  = name
        
        #Set bounds
        if hasattr(bounds, 'items'):
            self.bounds      = bounds['lb'], bounds['ub']
        else:
            self.bounds      = tuple(bounds)
    
        #Create priors
        prior_                        = ['uniform', *self.bounds] if prior is None else prior
        sample_                       = prior_ if sample is None else sample
        self.prior_type, self.prior   = self.read_prior(prior_,   scale)
        self.sample_type, self.sample = self.read_prior(sample_, scale, 'sample')
        
        #Set scale
        if scale not in self._scale:
            raise DunlinOptimizationError('scale', scale, list(self._scale.values()))
        self.scale_type = scale
        
    def scale(self, x):
        return self._scale[self.scale_type](x)
    
    def unscale(self, x):
        return self._unscale[self.scale_type](x)
    
    def get_prior(self, x):
        if self.prior_type in ['uniform', 'normal', 'laplace', 'logNormal', 'logLaplace']:
            x_ = self.unscale(x)
        else:
            x_ = x

        prior_value = self.prior.pdf(x_)
        return prior_value
    
    def get_opt_bounds(self):
        #This depends solely on the scale
        lb, ub = self.bounds
        return self.scale(lb), self.scale(ub)
            
    def __call__(self, x):
        return self.get_prior(x)
    
    def to_dict(self):
        return {'name'   : self.name,  
                'bounds' : self.bounds,     
                'prior'  : [self.prior_type, *self.prior.args], 
                'sample' : [self.sample_type, *self.sample.args],
                'scale'  : self.scale_type
                }
    
    def __repr__(self):
        return f'{type(self).__name__} {self.name}<bounds: {self.get_opt_bounds()}, scale: {self.scale_type}, prior: {self.prior_type}, sample: {self.sample_type}>'
    
    def __str__(self):
        return self.__repr__()
    
class OptResult():
    '''
    Note: Many optimization algorithms seek to MINIMIZE an objective function, 
    while Bayesian approaches attempt to MAXIMIZE the objective function.
    
    Therefore, the NEGATIVE of the log-likelihood should be used instead of the 
    regular log-likelihood for instantiation. The argument "to_minimize" is therefore
    stored under the attribute "neg_log_likelihood". For curve-fitting, this is 
    simply the SSE function.
    '''
    @classmethod
    def parameter_estimation(cls, model, dataset, **kwargs):
        get_SSE = ws.SSECalculator(model, dataset)
        return cls.from_model(model, get_SSE, **kwargs)
    
    @classmethod
    def from_model(cls, model, to_minimize=None, _attr='optim_args'):
        kwargs     = getattr(model, _attr, {})
        optimizers = {}
        if to_minimize is not None:
            if callable(to_minimize):
                to_minimize_ = to_minimize
            else:
                raise NotImplementedError('Still in the works.')
                to_minimize_ = model.get_exv(to_minimize)
            kwargs    = {**kwargs, **{'to_minimize': to_minimize_}}
        
        for estimate, nominal in model.params.to_dict('index').items():
            optimizer = cls(nominal=nominal, **kwargs)
            optimizers[estimate] = optimizer
        
        return optimizers
    
    def __init__(self, nominal, free_params, to_minimize, **kwargs):
        self.settings           = kwargs.get('settings', {})
        self.line_args          = kwargs.get('line_args', {})
        self.sampled_params     = []
        self.sampled_index      = np.zeros(len(free_params), dtype=np.int32)
        self.nominal            = np.array(list(nominal.values()))
        self.neg_log_likelihood = to_minimize
        self.names              = tuple(nominal.keys())
        self.opt_result         = {}
        
        c = 0
        
        for i, p in enumerate(nominal):
            if p in free_params:
                sampled_param = SampledParam(p, **free_params[p])
                self.sampled_params.append(sampled_param)
                self.sampled_index[c] = i
                
                c += 1
    def get_objective(self, free_params_array, _a=None, _p=None):
        priors = np.zeros(len(free_params_array))
        scaled = np.zeros(len(free_params_array))
        for i, (x, sp) in enumerate( zip(free_params_array, self.sampled_params) ):
            priors[i] = sp(x)
            scaled[i] = sp.unscale(x)
        
        params_array  = reconstruct(self.nominal, self.sampled_index, scaled)
        neg_posterior = self.neg_log_likelihood(params_array) - logsum(priors)
        
        if _a is not None:
            _a.append(params_array)
        if _p is not None:
            _p.append(neg_posterior)
            
        return neg_posterior
    
    def get_bounds(self):
        return [p.get_opt_bounds() for p in self.sampled_params]
        
    def differential_evolution(self, **kwargs):
        a        = []
        p        = []
        func     = self.get_objective 
        bounds   = self.get_bounds()
        settings = {**{'bounds': bounds}, **self.settings, **kwargs}
        result   = sop.differential_evolution(func, args=(a, p), **settings)
        cols     = [*self.names, '_posterior']
        p        = np.array(p)[:,np.newaxis]
        a        = np.concatenate((a, p), axis=1)
        opt_result = {'o': result, 
                      'a': pd.DataFrame(a, columns=cols)
                      }
        self.opt_result = opt_result
        return opt_result
    
    def __getitem__(self, key):
        try:
            df = self.opt_result['a']
        except:
            raise DunlinOptimizationError.no_opt_result()
        
        return df[key]
    
    def __getattr__(self, attr):
        if attr == 'posterior':
            return self.opt_result.get('a', None)
        elif attr in self.opt_result:
            return self.opt_result[attr]
        else:
            raise AttributeError(f'"{type(self).__name__}" does not have attribute "{attr}"')
    
    def __repr__(self):
        lst = ', '.join([sp.name for sp in self.sampled_params])
        return f'{type(self).__name__}<sampled_params: [{lst}]>'
    
    def __str__(self):
        return self.__repr__()
    
    def get_best(self, n=10):
        posterior = self.posterior.sort_values('_posterior').iloc[:n]
        
        best_params    = posterior.drop(columns=['_posterior']) 
        best_posterior = posterior['_posterior']
        
        return best_params, best_posterior
    
@njit
def logsum(arr):
    return np.sum(np.log(arr))

@njit
def reconstruct(nominal, sampled_index, scaled_free_params_array):
    params                = nominal.copy()
    params[sampled_index] = scaled_free_params_array
    
    return params

###############################################################################
#Plotting
###############################################################################    
def plot_all_opt_results(all_opt_results, AX, **line_args):
    AX1 = AX
    for model_key, opt_results in all_opt_results.items():
        AX_model = AX.get(model_key, None)
        
        if not AX_model:
            continue
        line_args_model = {k: v.get(model_key, {}) if type(v) == dict else v for k,v in line_args.items()}
        AX1[model_key]  = plot_opt_results(opt_results, AX_model, **line_args_model)
    return AX1
    
def plot_opt_results(opt_results, AX, palette=None, **line_args):
    global colors
    
    AX1       = AX
    for estimate, opt_result in opt_results.items(): 
        for var, ax_ in AX1.items():
            
            ax             = upp.recursive_get(ax_, estimate, var) 
            line_args_     = {**opt_result.line_args, **line_args}
            line_args_     = {k: upp.recursive_get(v, estimate, var) for k, v in line_args_.items()}
            
            #Process special keywords
            color = line_args_.get('color')
            if type(color) == str:
                line_args_['color'] = colors[color]
                
            label_scheme   = line_args_.get('label', 'scenario, estimate')
            if label_scheme == 'estimate':
                label = f'{estimate}'
            elif label_scheme == 'model_key':
                label = f'{opt_result.model_key}'
            elif label_scheme == 'model_key, estimate':
                label = f'{opt_result.model_key}, {estimate}'
            else:
                label = f'{estimate}'
            
            line_args_['label'] = label
            plot_type           = line_args_.get('plot_type', 'line')
            
            #Plot
            if plot_type == 'line':
                if line_args_.get('marker', None) and 'linestyle' not in line_args_:
                    line_args_['linestyle'] = 'None'
                
                if type(var) == tuple:
                    x_vals, y_vals = opt_result[var[0]].values, opt_result[var[1]].values
                    ax.plot(x_vals, y_vals, **line_args_)
                    ax.set_xlabel(var[0])
                    ax.set_ylabel(var[0])
                else:
                    y_vals = opt_result[var].values
                    ax.plot(y_vals, **line_args_)
                    ax.set_ylabel(var)
            else:
                raise ValueError(f'Unrecognized plot_type {plot_type}')
    
    return AX1

def plot_dataset(dataset, AX, yerr=None, xerr=None, **line_args):
    global colors
    
    AX1 = AX
    ye  = {} if yerr is None else yerr
    xe  = {} if xerr is None else xerr 
    
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
            y_err_ = ye.get((dtype, scenario, var))
            x_err_ = xe.get((dtype, scenario, var))
            
            for ax in ax_:
                ax.errorbar(x_vals, y_vals, y_err_, x_err_, **line_args_)
                ax.set_ylabel(var)
        else:
            raise ValueError(f'Unrecognized plot_type {plot_type}')
        
    return AX1
    