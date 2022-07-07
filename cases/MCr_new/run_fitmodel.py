import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

import addpath
import dunlin as dn
import cell_calculation as cc
import preprocess       as pp 
from fitmodel import *

def run(model_filename, medium, runs, save, data_line_args, posterior_line_args):
    
    #Set up the inputs
    mapping        = {(medium, 0) : 0, (medium, 1) : 1}
    loaded         = dn.load_file(model_filename)
    model          = loaded.parsed['Resource'] 
    expdata        = pp.trd0.reindex(['x', 'R', 'H', 'R_frac', 'H_frac', 'mu'], 
                                     mapping, 
                                     model=model,
                                     no_fit={'R_frac', 'H_frac', 'mu'}
                                     )
    cfdata = expdata
    
    cfdata.adjust_model_init(model, ['R', 'H', 'x'])
    adjust_yield(cfdata, model)

    #Evaluate the curvefitters
    if runs:
        curvefitters = dn.fit_model(model, 
                                    cfdata, 
                                    runs, 
                                    algo='simulated_annealing', 
                                    const_sd=True,
                                    x0_nominal=True
                                    )
        best         = dn.get_best_optimization(curvefitters)
        
    else:
        curvefitters = []
        best         = None
    
    #Plot the results
    fig, AX_lst, AX = make_AX(model, 
                              curvefitters, 
                              expdata, 
                              plot_guess=True,
                              data_line_args=data_line_args,
                              posterior_line_args=posterior_line_args
                              )
    
    
    fig, AX = make_prez_AX(model,
                           curvefitters, 
                           expdata, 
                           model_filename, 
                           medium, 
                           save=save,  
                           data_line_args=data_line_args,
                           posterior_line_args=posterior_line_args
                           )
    
    #Save the fitted parameters
    data_name  = cond.replace('.', '')
    model_name = model_filename.split('.')[0].split('_', 1)[1]
    
    write_dunl(model_name, data_name, best, model, )
               
    
    return curvefitters, best, fig, AX

if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    
    model_filename = 'curvefit_G{}.dunl'
    medium         = '0.4Glu'
    runs           = 1
    save           = True
    
    data_line_args      = dict(thin=2, capsize=5)
    posterior_line_args = dict(label='{scenario}')
    
    
    for i in [6]:#, 5, 4, 3, 3, 2, 1]:
        model_filename_ = model_filename.format(i)
        print(model_filename_)
        curvefitters, best, fig, AX = run(model_filename_, 
                                          medium, 
                                          runs, 
                                          save, 
                                          data_line_args, 
                                          posterior_line_args
                                          )
        print()