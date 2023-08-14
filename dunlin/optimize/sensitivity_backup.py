import numpy  as np
import pandas as pd
from collections   import namedtuple
from SALib.sample  import saltelli
from SALib.analyze import sobol

import dunlin.utils as ut

###############################################################################
#Function-Based
###############################################################################
def test_sensitivity(variable, model, algo='sobol', **kwargs):
    sensitivity_test = SensitivityTest(model)
    method = getattr(sensitivity_test, 'analyze_'+algo, None)
    
    if method is None:
        raise AttributeError(f'Could not algorithm for "{algo}"')
    
    sensitivity_result = method(variable, **kwargs)
    
    return sensitivity_result

def plot_sensitivity(sensitivity_result):
    raise NotImplementedError('Not implemented yet.')


###############################################################################
#Classes
###############################################################################
Result = namedtuple('SensivityResult', 'dtype Si Y samples problem')

class SensitivityTest:
    def __init__(self, model):
        self.ref   = model.ref
        self.model = model
        
        #Set up SALib arguments
        #Extracting the nominal and tspan now "freezes" the arguments
        #This helps to avoid side effects
        self.nominal          = model.parameter_dict
        self.tspan            = {k: v.copy() for k, v in model.tspan.items()}
        self.free_parameters  = model.optim_args['free_parameters']
        self.sensitivity_args = model.optim_args.get('sensitivity', {})
        
    @property
    def parameters(self):
        return self.model.parameters
    
    def get_groups(self):
        dct = self.sensitivity_args.get('groups')
        
        if not dct:
            return
        else:
            inverted = {}
            
            for group_name, lst in dct.items():
                for variable in lst:
                    if variable in inverted:
                        raise ValueError('Groups must be mutually exclusive.')
                    inverted[variable] = group_name
            
            free_parameters = set(self.free_parameters)
            
            difference = free_parameters.difference(inverted)
            if difference:
                raise ValueError(f'No group assigned for {difference}')
            
            return inverted
        
    @property
    def problem(self):
        free_parameters = self.free_parameters
        
        #Define SALib problem
        bounds = []
        for parameter, args in self.free_parameters.items():
            if ut.isdictlike(args):
                bounds_ = args['bounds']
            else:
                bounds_ = args[0]
            
            if len(bounds_) != 2:
                raise ValueError('Expected a pair in the form (lower_bounds, upper_bound)')
            elif bounds_[0] >= bounds_[1]:
                msg  = f'Invalid bounds for {parameter}: {args}.'
                msg += ' Ensure lower bound is lower than upper bound.'
                raise ValueError(msg)
                
            bounds.append(bounds_)
        
        #Collate problem
        problem = {'num_vars': len(free_parameters),
                   'names'   : list(free_parameters),
                   'bounds'  : bounds,
                   'dist'    : self.sensitivity_args.get('dist', 'unif')
                   }
        
        groups = self.get_groups()
        if groups:
            problem['groups'] = [groups[variable] for variable in problem['names']] 
        
        
        return problem
    
    ###########################################################################
    #Model Evaluation
    ###########################################################################
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    @staticmethod
    def _split(variable):
        #Determine which variables are actually callables
        if ut.islistlike(variable):
            vs = []
            cs = []
            
            for v in variable:
                if callable(variable):
                    cs.append(v)
                else:
                    vs.append(v)
        else:
            if callable(variable):
                cs = [variable]
                vs = []
            else:
                vs = [variable]
                cs = []
        
        return vs, cs
    
    def evaluate(self, 
                 variable, 
                 problem, 
                 free_parameter_samples, 
                 combine_scenarios=False
                 ):
        if combine_scenarios:
            return self._eval_combined(variable, problem, free_parameter_samples)
        else:
            return self._eval_separate(variable, problem, free_parameter_samples)
        
    def _eval_separate(self, variable, problem, free_parameter_samples):
        vs, cs    = self._split(variable)
        nominal   = self.nominal.copy()
        scenarios = nominal.index
        length    = free_parameter_samples.shape[0]
        tspan     = self.tspan
        
        V = np.zeros((len(vs), len(scenarios), length)) if vs else None
        C = {call: {c: np.zeros(length) for c in scenarios} for call in cs}
        
        #Iterate and evaluate
        for i, free_parameter_array in enumerate(free_parameter_samples):
            #For callables
            for call in cs:
                y = call(free_parameter_array)
                for scenario in scenarios:
                    C[call][i] = y[scenario]
            
            if not vs:
                continue
            
            #For actual variables
            nominal[problem['names']] = free_parameter_array
            
            for ii, (scenario, row) in enumerate(nominal.iterrows()):
                ir    = self(scenario=scenario, 
                             p0=row.values, 
                             tspan=tspan[scenario]
                             )
                
                #Update the array
                V[:, ii, i] += ir[vs]
        
        #Collate the results
        V = {v: {c: row for c, row in zip(scenarios, table) } for v, table in zip(vs, V)}
        Y = {**V, **C}
        
        return Y
        
    def _eval_combined(self, variable, problem, free_parameter_samples):
        
        vs, cs = self._split(variable)
        tspan  = self.tspan
        
        V = np.zeros((len(vs), free_parameter_samples.shape[0])) if vs else None
        C = {call: np.zeros(free_parameter_samples.shape[0]) for call in cs}
        
        #Iterate and evaluate
        nominal = self.nominal.copy()
        
        for i, free_parameter_array in enumerate(free_parameter_samples):
            #For callables
            for call in cs:
                y          = call(free_parameter_array)
                C[call][i] = y
            
            if not vs:
                continue
            
            #For actual variables
            nominal[problem['names']] = free_parameter_array
            
            for scenario, row in nominal.iterrows():
                ir = self(scenario=scenario, 
                          p0=row.values,
                          tspan=tspan[scenario]
                          )
                
                #Update the array
                V[:, i] += ir[vs]
        
        #Collate the results
        V = dict(zip(vs, V)) if vs else {}
        Y = {**V, **C}
        
        return Y
    
    ###########################################################################
    #Analysis
    ###########################################################################
    def sample_saltelli(self, **sampling_settings):
        settings = self.sensitivity_args.get('sample', {})
        problem  = self.problem
        default  = {'N' : 1024}
        
        settings = {**default, **settings, **sampling_settings}
        
        #Define samples for SALib
        free_parameter_samples = saltelli.sample(problem, **settings)
        
        return problem, free_parameter_samples
    
    def analyze_sobol(self, variable, combine_scenarios=True, 
                      sampling_settings=None, analysis_settings=None
                      ):
        #Make samples
        sampling_settings = sampling_settings if sampling_settings else {}
            
        problem, free_parameter_samples = self.sample_saltelli(**sampling_settings)
        
        #Evalute model
        Y = self.evaluate('final_x1', 
                          problem, 
                          free_parameter_samples, 
                          combine_scenarios
                          )
        
        #Run sobol
        if analysis_settings:
            analysis_settings = {**self.sensitivity_args.get('analyze', {}),
                                 **analysis_settings
                                 }
        else:
            analysis_settings = self.sensitivity_args.get('analyze', {}).copy()
        
        analysis_settings['print_to_console'] = analysis_settings.pop('disp', True)
        
        Si = {}
        for v, Y_ in Y.items():
            if combine_scenarios:
                Si_ = sobol.analyze(problem, 
                                    Y_, 
                                    **analysis_settings
                                    )

            else:
                Si_   = {}
                for scenario in Y_:
                    Si_[scenario] = sobol.analyze(problem, 
                                                  Y_[scenario], 
                                                  **analysis_settings
                                                  )
                    
            Si_   = Result('sobol', Si_, Y_, free_parameter_samples, problem)
            Si[v] = Si_
        
        #Extract if required and then return
        if not ut.islistlike(variable):
            Si = Si[variable]
        return Si
    
    