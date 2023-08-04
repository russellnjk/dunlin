import numpy             as     np
import pandas            as     pd
from numba               import njit
from scipy.stats         import norm, laplace, lognorm, loglaplace, uniform

###############################################################################
#Sampled Parameters
###############################################################################
class SampledParam:
    '''A class for sampling parameter values. Definition based on PETab. Not 
    meant to be mutable.
    '''
    _scale = {'lin'   : lambda x: x,
              'log'   : lambda x: np.log(x),
              'log10' : lambda x: np.log10(x)
              }
    
    _unscale = {'lin'   : lambda x: x,
                'log'   : lambda x: np.exp(x),
                'log10' : lambda x: 10**x
                }
    
    _priors = {'uniform'    : lambda lb, ub    : 1,
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
    _priors['psuni']   = _priors['uniform']
    _priors['psnorm']  = _priors['normal']
    _priors['pslap']   = _priors['laplace']

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
            raise InvalidPrior(_name, prior, list(cls._priors.keys()))
        except:
            raise InvalidPrior(_name, prior, 'list/tuple in the order [type, loc, scale] or dict with keys type, loc and scale')
        return ptype, func(a, b)
            
    def __init__(self, 
                 name, 
                 bounds, 
                 prior        = None, 
                 sample       = None, 
                 scale        = 'lin', 
                 guess        = None, 
                 scaled_guess = False):
        #Set name
        self.name  = name
        
        #Set bounds
        if hasattr(bounds, 'items'):
            self.bounds      = bounds['lb'], bounds['ub']
        else:
            self.bounds      = tuple(bounds)
        if self.bounds[0] >= self.bounds[1] or len(self.bounds) != 2:
            raise InvalidBoundsError(name, self.bounds)
        
        #Create priors
        prior_                             = ['uniform', *self.bounds] if prior is None else prior
        sample_                            = prior_ if sample is None else sample
        self.prior_type, self.prior_calc   = self.read_prior(prior_,   scale)
        self.sample_type, self.sample_calc = self.read_prior(sample_, scale, 'sample')
        
        #Set scale
        if scale not in self._scale:
            msg = f'Invalid scale. Must be one of: {list(self._scale.keys())}\nReceived: {scale}'
            raise ScaleError(msg)
        self.scale_type = scale
        
        #Set guess. Set underlying attr first then use setter.
        self._guess = None
        self.set_guess(guess, scaled=scaled_guess)
    
    def get_guess(self, scaled=True):
        if self._guess is None:
            lb, ub = self.get_opt_bounds() if scaled else self.bounds
            guess  = (ub+lb)/2
        else:
            guess = self.scale(self._guess) if scaled else self._guess
    
        return guess
    
    def set_guess(self, value, scaled=True):
        if value is None:
            self._guess = None
            return
        
        try:
            value = float(value)
        except:
            raise InvalidGuessError('Could not convet guess into a float.')
        
        lb, ub = self.get_opt_bounds() if scaled else self.bounds
        if value < lb or value > ub:
            msg0 = f'Attempted to set guess attribute of {type(self).__name__} with a value outside bounds.'
            msg1 = '\nlb={}, ub={}, guess={}, scaled={}'
        
            raise InvalidGuessError(msg0+msg1.format(lb, ub, value, scaled))
        else:
            self._guess = self.unscale(value) if scaled else value
    
    def scale(self, x):
        return self._scale[self.scale_type](x)
    
    def unscale(self, x):
        return self._unscale[self.scale_type](x)
    
    def get_prior(self, x):
        if self.prior_type == 'uniform':
            return 1
        
        if self.prior_type in ['normal', 'laplace', 'logNormal', 'logLaplace']:
            x_ = self.unscale(x)
        else:
            x_ = x

        prior_value = self.prior_calc.pdf(x_)
        return prior_value
    
    def get_opt_bounds(self):
        #This depends solely on the scale
        lb, ub = self.bounds
        return self.scale(lb), self.scale(ub)
            
    def __call__(self, x):
        return self.get_prior(x)
    
    def new_sample(self):
        return self.sample_calc.rvs()

    def to_dict(self):
        return {'name'   : self.name,  
                'bounds' : self.bounds,     
                'prior'  : [self.prior_type, *self.prior.args], 
                'sample' : [self.sample_type, *self.sample.args],
                'scale'  : self.scale_type
                }
    
    def __repr__(self):
        return str(self)
    
    def __str__(self):
        return f'{type(self).__name__} {self.name}<bounds: {self.get_opt_bounds()}, scale: {self.scale_type}, prior: {self.prior_type}, sample: {self.sample_type}>'

class ScaleError(Exception):
    pass

class InvalidGuessError(Exception):
    pass

class InvalidPrior(Exception):
    def __init__(self, arg, value, correct):
        super().__init__(f'Invalid {arg} format: {value}\nValue must be {correct} ')

class InvalidBoundsError(Exception):
    def __init__(cls, name, bounds):
        super().__init__(f'Invalid bounds given for {name}: {bounds}')

###############################################################################
#Bounds
###############################################################################    
class Bounds:
    '''A class for bounds testing. In Scipy, the test is called AFTER evaluation 
    of the objective function. This class is not meant to be mutable.
    '''
    def __init__(self, sampled_params):
        self._xmax = np.zeros(len(sampled_params))
        self._xmin = np.zeros(len(sampled_params))
        self._idx  = {}
        
        for i, sp in enumerate(sampled_params):
            self._xmax[i] = sp.scale(sp.bounds[1])
            self._xmin[i] = sp.scale(sp.bounds[0])
            self._idx[i]  = sp
        
        
    def __call__(self, **kwargs):
        '''Defined according to Scipy to ensure compatibility.Signature is:
        accept_test(f_new=f_new, x_new=x_new, f_old=fold, x_old=x_old)
        '''
        #Follow scipy definition in order to ensure compatibility with scipy algos
        xnew = kwargs['x_new']
        xmax = self._xmax
        xmin = self._xmin
        
        return self._check_bounds(xnew, xmax, xmin)
    
    @staticmethod
    @njit
    def _check_bounds(xnew, xmax, xmin):
        tmax = np.all(xnew < xmax)
        tmin = np.all(xnew > xmin)
        
        return tmax and tmin
    
    def __repr__(self):
        return str(self)
    
    def __str__(self):
        return f'{type(self).__name__}({self._xmin}, {self._xmax})'
    
    def get_out_of_bounds(self, **kwargs):
        xnew = kwargs['x_new']
        xmax = self._xmax
        xmin = self._xmin
        
        over  = {}
        under = {}
        
        for i, (val, bnd) in enumerate(zip(xnew, xmax)):
            if bnd <= val:
                sp            = self._idx[i]
                over[sp.name] = sp.unscale(bnd), sp.unscale(val)
        
        for i, (val, bnd) in enumerate(zip(xnew, xmin)):
            if bnd >= val:
                sp             = self._idx[i]
                under[sp.name] = sp.unscale(bnd), sp.unscale(val)
        
        return over, under
    
    def to_pairs(self):
        xmax = self._xmax
        xmin = self._xmin
        
        return [*zip(xmin, xmax)]
        
###############################################################################
#Dunlin Errors
###############################################################################    
class DunlinOptimizationError(Exception):
    @classmethod
    def raise_template(cls, msg):
        return cls(msg)
    
    @classmethod
    def no_opt_result(cls):
        return cls.raise_template('No optimization yet. Make sure you have run one of the optimization algorithms.')
    
    @classmethod
    def nominal(cls):
        return cls.raise_template('When instantiating an OptResult object, make sure the nominal is a DataFrame of dict that can be converted into a DataFrame.')
    
    @classmethod
    def no_algo(cls, algo):
        return cls.raise_template(f'No algorithm called "{algo}".')
    
    @classmethod
    def no_optim_args(cls, model_name):
        return cls.raise_template(f'The optim_args attribute for Model{model_name} has not been set.')
    