import numpy  as np
from numbers import Number
from SALib   import ProblemSpec
from typing  import Callable

Parameter = str

class SensitivityMixin:
    '''To be mixed into Optimizer.
    '''
    settings        : dict
    free_parameters : dict[Parameter, dict] 
    get_objective   : Callable
    
    def get_objective_nd(self, free_parameters_array: np.ndarray) -> Number:
        if len(free_parameters_array) == 1:
            return self.get_objective(free_parameters_array)
        
        neg_posterior = np.zeros(len(free_parameters_array))
        
        for i, row in enumerate(free_parameters_array):
            neg_posterior[i] = self.get_objective(row)
       
        return neg_posterior
    
    @property
    def salib_problem(self) -> dict:
        '''Generates a problem according to SALib specifications. Note that 
        the bounds are given as scaled values. E.g. If a sampled parameter's 
        `scale` attribute is `"log10"`, then its bounds are power of ten.
        '''
        sampled_parameters = self.sampled_parameters
        
        #Define SALib problem
        n      = 0
        names  = []
        bounds = []
        
        for sampled_parameter in sampled_parameters:
            n += 1
            names.append(sampled_parameter.name)
            
            #Get the bounds
            sample_type = sampled_parameter.sample_type
            if sample_type == 'uniform':
                lb, ub = sampled_parameter.scaled_bounds
                
            elif sample_type in {'normal', 'lognormal'}:
                
                mean = sampled_parameter.sample_calc.mean()
                sd   = sampled_parameter.sample_calc.std()
                lb   = mean - 3*sd
                ub   = mean + 3*sd
                lb   = sampled_parameter.scale(lb)
                ub   = sampled_parameter.scale(ub)
            
            else:
                
                mean, variance = sampled_parameter.sample_calc.stats()
                
                lb = mean - 3*variance**.5*3
                ub = mean + 3*variance**.5*3
                
            
            bounds.append([lb, ub])
                
        #Collate problem
        problem = {'num_vars': n,
                   'names'   : names,
                   'bounds'  : bounds,
                   }
        
        return problem
    
    
    def run_sobol(self, 
                  N : int  = 1024,
                  ):
        #Parse kwargs
        kwargs   = self.settings.get('sobol', {})
        
        #Generate samples
        problem           = self.salib_problem
        calc_second_order = kwargs.get('calc_second_order', True)
        
        #Make ProblemSpec and chain methods
        sp = ProblemSpec(problem)

        (sp
          .sample_sobol(N, calc_second_order=calc_second_order)
          .evaluate(self.get_objective_nd)
          .analyze_sobol(**kwargs)
          )
        return sp
        
    def run_dgsm(self, 
                 N: int = 1000
                 ):
        #Parse kwargs
        kwargs   = {**self.settings.get('dgsm', {})}
        
        #Generate samples
        problem           = self.salib_problem
        
        sp = ProblemSpec(problem)
        
        (sp
         .sample_finite_diff(N           = N, 
                             delta       = kwargs.pop('delta', 0.01),
                             seed        = kwargs.pop('seed', None),
                             skip_values = kwargs.pop('skip_values', 1024)
                             )
         .evaluate(self.get_objective_nd)
         .analyze_dgsm()
         )
        
        return sp
       