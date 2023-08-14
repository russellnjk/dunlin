import numpy as np
from numba       import njit
from numbers     import Number
from scipy.stats import norm, laplace, lognorm, loglaplace
from typing      import Literal

###############################################################################
#Sampled Parameters
###############################################################################
class SampledParam:
    '''A class for sampling parameter values. Definition based on PETab. Not 
    meant to be mutable.
    '''
    #Functions for scaling and prior calculation
    _scale = {'lin'   : lambda x: x,
              'log'   : lambda x: np.log(x),
              'log10' : lambda x: np.log10(x)
              }
    
    _unscale = {'lin'   : lambda x: x,
                'log'   : lambda x: np.exp(x),
                'log10' : lambda x: 10**x
                }
    
    _priors = {'normal'     : lambda mean, sd  : norm(mean, sd),
               'laplace'    : lambda loc, scale: laplace(loc, scale),
               'lognormal'  : lambda mean, sd  : lognorm(mean, sd),
               'loglaplace' : lambda loc, scale: loglaplace(loc, scale),
               }
    
    _priors['parameterScaleNormal' ] = _priors['normal' ]
    _priors['parameterScaleLaplace'] = _priors['laplace' ]
            
    def __init__(self, 
                 name, 
                 bounds       : tuple[Number, Number], 
                 prior        : dict = None, 
                 sample       : dict = None, 
                 scale        : Literal['lin', 'log', 'log10'] = 'lin', 
                 guess        : np.ndarray = None, 
                 scaled_guess : bool = False):
        #Set name
        self.name  = name
        
        #Set bounds. Use absolute values.
        match bounds:
            case {'lb': lb, 'ub': ub} if len(bounds) == 2:
                self.bounds = lb, ub
                self.lb     = lb
                self.ub     = ub
            case [lb, ub]:
                self.bounds = lb, ub
                self.lb     = lb
                self.ub     = ub
            case _:
                msg  = f'Invalid bounds provided for sampled parameter {name}. '
                msg += f'Received {bounds}.'
                raise ValueError(msg)
        
        #Set scale
        if scale not in self._scale:
            msg = f'Invalid scale. Must be one of: {list(self._scale.keys())}\nReceived: {scale}'
            raise ValueError(msg)
        self.scale_type = scale
        self.scale      = self._scale[scale]
        self.unscale    = self._unscale[scale]
        
        #Set the function for calculating the prior
        match prior:
            case None:
                self.prior_type = 'uniform'
                self.prior_calc = None
            
            case ['uniform', lb, ub] :
                msg  = f'Error in instantiating sampled parameter {name}. '
                msg += 'The keyword argument "prior" can only be used for non-uniform priors.'
                raise ValueError(msg)
            
            case [prior_type, arg0, arg1]:
                self.prior_type = prior_type
                self.prior_calc = self._priors[prior_type](arg0, arg1)
            
            case {'type': 'uniform'}:
                msg  = f'Error in instantiating sampled parameter {name}. '
                msg += 'The keyword argument "prior" can only be used for non-uniform priors.'
                raise ValueError(msg)
            
            case {'type' : 'normal'|'lognormal'|'parameterScaleNormal', 
                  'mean' : mean, 
                  'sd'   : sd, 
                  **rest
                  }:
                if rest:
                    msg  = f'Error in instantiating sampled parameter {name}. '
                    msg += 'Laplace/loglaplace priors cannot have keys other than "mean" and "sd". '
                    msg += 'Extra keys: {list(rest)}.'
                    raise ValueError(msg)
                
                prior_type      = prior['type']
                self.prior_type = prior_type
                self.prior_calc = self._priors[prior_type](mean, sd)
            
            case {'type' : 'laplace'|'loglaplace'|'parameterScaleLaplace', 
                  'loc'  : loc, 
                  'scale': scale, 
                  **rest
                  }:
                if rest:
                    msg  = f'Error in instantiating sampled parameter {name}. '
                    msg += 'Laplace/loglaplace priors cannot have keys other than "loc" and "scale". '
                    msg += 'Extra keys: {list(rest)}.'
                    raise ValueError(msg)
                
                prior_type      = prior['type']
                self.prior_type = prior_type
                self.prior_calc = self._priors[prior_type](loc, scale)
            
            case _:
                msg  = f'Error in instantiating sampled parameter {name}. '
                msg += 'Invalid format for the keyword argument "prior". '
                msg += 'Uniform priors must not use this keyword argument. '
                msg += 'For normal/lognormal priors the correct format is [<prior_type>, <mean>, <sd>]. '
                msg += 'For laplace/loglaplace priors the correct format is [<prior_type>, <loc>, <scale>]. '
                msg += 'Received {prior}.'
                
                raise ValueError(msg)
                
        #Set the function for sampling the prior
        match sample:
            case None:
                self.sample_type = 'uniform'
                self.sample_calc = None
            
            case ['uniform', lb, ub] :
                msg  = f'Error in instantiating sampled parameter {name}. '
                msg += 'The keyword argument "sample" can only be used for non-uniform samples.'
                raise ValueError(msg)
            
            case [sample_type, arg0, arg1]:
                self.sample_type = sample_type
                self.sample_calc = self._priors[sample_type](arg0, arg1)
            
            case {'type': 'uniform'}:
                msg  = f'Error in instantiating sampled parameter {name}. '
                msg += 'The keyword argument "sample" can only be used for non-uniform samples.'
                raise ValueError(msg)
            
            case {'type' : 'normal'|'lognormal'|'parameterScaleNormal', 
                  'mean' : mean, 
                  'sd'   : sd, 
                  **rest
                  }:
                if rest:
                    msg  = f'Error in instantiating sampled parameter {name}. '
                    msg += 'Laplace/loglaplace samples cannot have keys other than "mean" and "sd". '
                    msg += 'Extra keys: {list(rest)}.'
                    raise ValueError(msg)
                
                sample_type      = sample['type']
                self.sample_type = sample_type
                self.sample_calc = self._priors[sample_type](mean, sd)
            
            case {'type': 'laplace'|'loglaplace', 'loc': loc, 'scale': scale, **rest}:
                if rest:
                    msg  = f'Error in instantiating sampled parameter {name}. '
                    msg += 'Laplace/loglaplace samples cannot have keys other than "loc" and "scale". '
                    msg += 'Extra keys: {list(rest)}.'
                    raise ValueError(msg)
                
                sample_type      = sample['type']
                self.sample_type = sample_type
                self.sample_calc = self._priors[sample_type](loc, scale)
            
            case _:
                msg  = f'Error in instantiating sampled parameter {name}. '
                msg += 'Invalid format for the keyword argument "sample". '
                msg += 'Uniform samples must not use this keyword argument. '
                msg += 'For normal/lognormal samples the correct format is [<sample_type>, <mean>, <sd>]. '
                msg += 'For laplace/loglaplace samples the correct format is [<sample_type>, <loc>, <scale>]. '
                msg += 'Received {sample}.'
                
                raise ValueError(msg)
        
        # if prior is None:
        #     self.prior_type = 'uniform'
        #     self.prior_calc = self._priors['uniform'](self.lb, self.ub)
        # elif prior == 'uniform':
        #     self.prior_type = 'uniform'
        #     self.prior_calc = self._priors['uniform'](self.lb, self.ub)
        # elif prior in self._priors:
        #     if prior == 'uniform':
        #         self.prior_type = 'uniform'
        #         self.prior_calc = self._priors['uniform'](self.lb, self.ub)
        #     else:
        #         self.prior_type = prior
        #         self.prior_calc = self._priors[prior](self.)
            
        
        # #Create priors
        # prior_                             = ['uniform', *self.bounds] if prior is None else prior
        # sample_                            = prior_ if sample is None else sample
        # self.prior_type, self.prior_calc   = self.read_prior(prior_,   scale)
        # self.sample_type, self.sample_calc = self.read_prior(sample_, scale, 'sample')
        
        
        
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
            msg = 'Could not convet guess into a float.'
            raise ValueError(msg)
            
        lb, ub = self.get_opt_bounds() if scaled else self.bounds
        if value < lb or value > ub:
            msg0 = f'Attempted to set guess attribute of {type(self).__name__} with a value outside bounds.'
            msg1 = f'\nlb={lb}, ub={ub}, guess={value}, scaled={scaled}'
            msg  = msg0 + msg1
            
            raise ValueError(msg)
        else:
            self._guess = self.unscale(value) if scaled else value
    
    @property
    def scaled_bounds(self) -> tuple[Number, Number]:
        #This depends solely on the scale
        lb, ub = self.bounds
        return self.scale(lb), self.scale(ub)
            
    def __call__(self, x) -> Number:
        return self.get_prior(x)
    
    def get_prior(self, scaled_x: Number) -> Number:
        prior_type = self.prior_type
        
        #Case 0: The prior is uniform
        if prior_type == 'uniform':
            return 1
        
        #Case 1: The prior was defined in linear units
        elif prior_type in {'normal', 'lognormal', 'laplace', 'loglaplace'}:
            unscaled_x = self.unscale(scaled_x)
            return self.prior_calc.pdf(unscaled_x)
        
        #Case 2: The prior was defined in scaled units
        else:
            return self.prior_calc.pdf(scaled_x)
        
    def new_sample(self) -> np.ndarray:
        return self.sample_calc.rvs()

    def to_dict(self) -> dict:
        return {'name'   : self.name,  
                'bounds' : self.bounds,     
                'prior'  : [self.prior_type, *self.prior.args], 
                'sample' : [self.sample_type, *self.sample.args],
                'scale'  : self.scale_type
                }
    
    def __repr__(self):
        return str(self)
    
    def __str__(self):
        return f'{type(self).__name__} {self.name}<bounds: {self.scaled_bounds}, scale: {self.scale_type}, prior: {self.prior_type}, sample: {self.sample_type}>'

###############################################################################
#Bounds
###############################################################################    
class Bounds:
    '''A class for bounds testing. In Scipy, the test is called AFTER evaluation 
    of the objective function. This class is not meant to be mutable.
    '''
    def __init__(self, sampled_params: list[str]):
        self._xmax = np.zeros(len(sampled_params))
        self._xmin = np.zeros(len(sampled_params))
        self._idx  = {}
        
        for i, sp in enumerate(sampled_params):
            self._xmax[i] = sp.scale(sp.bounds[1])
            self._xmin[i] = sp.scale(sp.bounds[0])
            self._idx[i]  = sp
        
        
    def __call__(self, **kwargs):
        '''Defined according to Scipy to ensure compatibility. Signature is:
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
      