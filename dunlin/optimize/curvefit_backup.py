import numpy as np
from collections import namedtuple

###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin.simulate           as sim
import dunlin.utils              as ut 
import dunlin.optimize.wrap_SSE  as ws
import dunlin.optimize.optimizer as opt
import dunlin.data               as ddt

###############################################################################
#High-Level Algorithms
###############################################################################
def fit_model(model, dataset, runs=1, algo='differential_evolution', 
              const_sd=False, **kwargs
              ):
    curvefitters = []
    curvefitter  = Curvefitter(model, dataset, const_sd=const_sd)
    
    for i in range(runs):
        if i > 0:
            curvefitter = curvefitter.seed(i)
        
        method = getattr(curvefitter, 'run_' + algo, None)
        if method is None:
            raise opt.DunlinOptimizationError.no_algo(algo)
            
        method(**kwargs)
        
        curvefitters.append(curvefitter)
        
    return curvefitters 

###############################################################################
#Plotting
###############################################################################
def plot_curvefit(AX_line,/, 
                  curvefitters=None, 
                  dataset  =None, 
                  model=None, 
                  guess_line_args=None, 
                  data_line_args=None, 
                  posterior_line_args=None,
                  plot_guess=True, n=0
                  ):
    
    #Plot guess
    guess_plot_result = {}
    guess_sim_result  = None
    if plot_guess and model is not None:
        #Set up the plotting args
        guess_line_args = {} if guess_line_args is None else dict(guess_line_args)
        guess_line_args.setdefault('linestyle', ':')
        
        if curvefitters:    
            guess_line_args.setdefault('label', '_nolabel')
        
        guess_line_args.update(model.optim_args.get('line_args', {}))
        
        #Simulate and plot
        sim_result = sim.simulate_model(model)
        sim.plot_line(AX_line, sim_result, **guess_line_args)
        
        #Update the result
        guess_sim_result = sim_result
    
    #Plot data
    data_plot_result = {}
    data_line_args   = {} if data_line_args is None else dict(data_line_args)
    dataset_         = None
    
    if dataset is not None:
        #Set up the plotting args
        if plot_guess or curvefitters:
            data_line_args['label'] = '_nolabel'
        
        #Generate dataset and plot
        if type(dataset) == ddt.TimeResponseData:
            dataset_ = dataset
        elif model is None:
            msg = 'A model must be provided as if dataset is neither None nor dunlin.Dataset.'
            raise ValueError(msg)
        else:
            dataset_ = ddt.TimeResponseData(*dataset, model=model)
        
        sim.plot_line(AX_line, dataset_, **data_line_args)
        
    #Plot curvefitting result
    curvefitters          = [] if not curvefitters else curvefitters
    posterior_plot_result = {}
    posterior_sim_result  = []
    
    #Set up plot args
    posterior_line_args = {} if posterior_line_args is None else dict(posterior_line_args)
    
    #Iterate and plot
    for cft in curvefitters:
        sim_results = cft.simulate(n)
        
        if ut.isnum(n):
            sim_results = [sim_results]
        
        temp = sim.plot_line(AX_line, sim_results, **posterior_line_args)
      
        #Update
        posterior_sim_result.append(temp)
    
    #Collate the results
    result = {'posterior_plot_result' : posterior_plot_result, 
              'guess_plot_result'     : guess_plot_result, 
              'data_plot_result'      : data_plot_result, 
              'posterior_sim_result'  : posterior_sim_result,
              'dataset_'              : dataset_,
              'guess_sim_result'      : guess_sim_result
              }
    
    return result

###############################################################################
#Curvefitter
###############################################################################
class Curvefitter(opt.Optimizer):
    def __init__(self, model, dataset, **kwargs):
        if type(dataset) == ddt.TimeResponseData:
            sse_calc = ws.SSECalculator(model, dataset, **kwargs)
        else:
            sse_calc = ws.SSECalculator(model, *dataset, **kwargs)
        
        #Call superclass constructor
        nominal         = model.parameters
        free_parameters = model.optim_args.get('free_parameters', {})
        settings        = model.optim_args.get('settings',    {})
        trace_args      = model.optim_args.get('trace_args',  {})
        
        super().__init__(nominal, free_parameters, sse_calc, settings, trace_args)
        
        #Add subclass attributes
        self.model = model
    
    ###########################################################################
    #Access and Modification
    ###########################################################################
    @property
    def sse_calc(self):
        return self.neg_log_likelihood
    
    ###########################################################################
    #Simulation
    ###########################################################################
    def simulate(self, n=0):
        parameters, *_ = self.trace.get_best(n)
        array          = parameters.values 
        
        if len(array.shape) == 1:
            p_dict     = self.sse_calc.reconstruct(array)
            sim_result = sim.simulate_model(self.model, p0=p_dict)
            
            return sim_result
        
        else:
            sim_results = []
            for fpa in array:
                p_dict     = self.sse_calc.reconstruct(fpa)
                sim_result = sim.simulate_model(self.model, p0=p_dict)
                
                sim_results.append(sim_result)
        
            return sim_results
    