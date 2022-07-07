import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

import addpath
import dunlin                as dn
import dunlin.datastructures as dst
import cell_calculation      as cc
import preprocess            as pp 

def make_AX(model, curvefitters, dataset, **kwargs):
    fig, AX_ = dn.figure(3, 6)

    AX                 = dict(zip(model.state_names, AX_))
    AX['R_frac']       = AX.pop('R')
    AX['H_frac']       = AX.pop('H')
    AX['jH']           = AX_[-9]
    AX['allprot']      = AX_[-7]
    AX['regR']         = AX_[-6]
    AX['mu']           = AX_[-5]
    AX['syn_R']        = AX_[-4]
    AX['A']              = AX_[-3]   
    AX[('mu', 'R_frac')] = AX_[-2]
    AX[('mu', 'syn_H')] = AX_[-1]
    cc.plot_R_vs_mu(AX_[-2])
    cc.plot_synH_vs_mu(AX_[-1])
    AX[('mu', 'R_frac')].set_xlim(0, 0.02)
    AX[('mu', 'R_frac')].set_ylim(0, 0.3)
    AX[('mu', 'syn_H')].set_xlim(0, 0.02)
    AX[('mu', 'syn_H')].set_ylim(0, 5e-3)
    
    
    dn.plot_curvefit(AX, 
                     curvefitters, 
                     expdata, 
                     model, 
                     **kwargs
                     )
    
    AX_[0].legend()
    
    return fig, AX_, AX

def make_prez_AX(model, curvefitters, dataset, model_filename, cond, save=True, path='', **kwargs):
    #For presentation plots
    fig, AX_ = dn.figure(1, 4, figsize=(12, 2.7), 
                         top=0.88, bottom=0.18, 
                         left=0.036, right=0.996,
                         wspace=0.28
                         )
    AX = {}
    AX['x']      = AX_[0]
    AX['R_frac'] = AX_[1]
    AX['H_frac'] = AX_[2]
    AX[('mu', 'R_frac')]     = AX_[3]
    cc.plot_R_vs_mu(AX_[3])
    AX[('mu', 'R_frac')].set_xlim(0, 0.025)
    AX[('mu', 'R_frac')].set_ylim(0, 0.3)
    
    dn.plot_curvefit(AX, 
                     curvefitters, 
                     dataset, 
                     model, 
                     plot_guess=False, 
                     **kwargs
                     )
    
    AX['x'].set_title('(A) x', pad=10)
    AX['R_frac'].set_title('(B) ϕR', pad=10)
    AX['H_frac'].set_title('(C) ϕH', pad=10)
    AX[('mu', 'R_frac')].set_title('(D) ϕR vs λ', pad=10)
    AX_[0].legend(loc='lower right', title='IPTG (mM)')
    
    if save:
        if path:
            fig.savefig(path)
        else:
            data_name  = cond.replace('.', '')
            model_name = model_filename.split('.')[0].split('_', 1)[1]
            new_filename = 'Model_' + model_name + '_' + data_name + '.png'
            fig.savefig(new_filename)
        
    return fig, AX

def make_prez_AX2(model, optrs, dataset, model_filename, cond, save=True, path='', **kwargs):
    
    fig, AX_ = dn.figure(1, 2, figsize=(12, 2.7), 
                         top=0.82, bottom=0.18, 
                         left=0.04, right=0.996,
                         wspace=0.24
                         )
    AX = {}
    AX['Q']   = AX_[0]
    AX['RmQ'] = AX_[1]
    
    dn.plot_curvefit(AX, 
                     curvefitters, 
                     dataset, 
                     model, 
                     plot_guess=False, 
                     **kwargs
                     )
    AX['Q'].set_title('(E) ϕQ', pad=20)
    AX['RmQ'].set_title('(F) RmQ', pad=20)
    AX_[0].legend(loc='upper left')
    
    if save:
        if path:
            fig.savefig(path, dpi=1000)
        
        else:
            data_name  = cond.replace('.', '')
            model_name = model_filename.split('.')[0].split('_', 1)[1]
            new_filename = 'Model_' + model_name + '_' + data_name + '_ax2.png'
            fig.savefig(new_filename, dpi=1000)
        
    return fig, AX

def write_dunl(model_name, data_name, best, model, filename='cf_result.dunl'):
    if best is None:
        return
    
    objective       = best.objective
    best_parameters = best.parameters
    n_free          = len(best_parameters)
    
    df                        = model.parameters
    df[best.parameters.index] = best_parameters
    
    data = {'objective'  : objective,
            'n_free'     : n_free,
            'parameters' : dst.ParameterDict(df, set())#C(df)
            }
    
    all_data = {model_name: {data_name: data}}
    
    dn.write_dunl_file(all_data, filename, op='merge')
    
def get_from_dunl(model_name, data_name, filename='cf_result.dunl'):
    all_data = dn.read_file(filename)
    data     = all_data[model_name][data_name]
    
    return data, data['parameters']
    
def adjust_yield(dataset, model):
    df = model.parameters
    for scenario, series in dataset['x']['x'].items():
        gb    = series.groupby(level=1)
        init  = gb.first().mean()
        final = gb.last().mean()
        
        dx = final-init
        Yield = dx/1000*(1e6/110)
        print('Yield', Yield)
        df.loc[scenario, 'Yield'] = Yield
    
    model.parameters = df
    
if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    
    model_filename = 'curvefit_G6.dunl'
    medium         = '0.4Glu'
    runs           = 1
    save           = True
    
    data_line_args      = dict(thin=2, capsize=5)
    posterior_line_args = dict(label='{scenario}')
    
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
    
    write_dunl('Test model', 'Test data', best, model, )
               