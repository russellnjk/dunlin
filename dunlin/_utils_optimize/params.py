import numpy             as     np
import pandas            as     pd
from numba               import njit
from scipy.stats         import norm, laplace, lognorm, loglaplace, uniform

###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin._utils_model.base_error as dbe

###############################################################################
#Classes for Parameters
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
        if self.bounds[0] >= self.bounds[1]:
            raise DunlinOptimizationError.invalid_bounds(name, self.bounds)
        
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
    
    def new_sample(self):
        return self.prior.rvs()

class Bounds():
    def __init__(self, sampled_params):
        self.xmax = np.zeros(len(sampled_params))
        self.xmin = np.zeros(len(sampled_params))
        
        for i, sp in enumerate(sampled_params):
            self.xmax[i] = sp.scale(sp.bounds[1])
            self.xmin[i] = sp.scale(sp.bounds[0])
    
    def __call__(self, **kwargs):
        xnew = kwargs['x_new']
        xmax = self.xmax
        xmin = self.xmin
        
        return self._check_bounds(xnew, xmax, xmin)
    
    @staticmethod
    @njit
    def _check_bounds(xnew, xmax, xmin):
        tmax = np.all(xnew <= xmax)
        tmin = np.all(xnew >= xmin)
        
        return tmax and tmin

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
    def invalid_bounds(cls, name, bounds):
        return cls.raise_template(f'Invalid bounds given for {name}: {bounds}', 2)
    
    @classmethod
    def no_opt_result(cls):
        return cls.raise_template('No optimization yet. Make sure you have run one of the optimization algorithms.', 10)
    
    @classmethod
    def nominal(cls):
        return cls.raise_template('When instantiating an OptResult object, make sure the nominal is a DataFrame of dict that can be converted into a DataFrame.', 10)
    
    @classmethod
    def no_algo(cls, algo):
        return cls.raise_template(f'No algorithm called "{algo}".', 12)
    
    @classmethod
    def no_optim_args(cls, model_key):
        return cls.raise_template(f'The optim_args attribute for Model{model_key} has not been set.', 13)
    