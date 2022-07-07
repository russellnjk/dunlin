import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

import addpath
import dunlin as dn
import dunlin.dataparser as dtp
import cell_calculation  as cc

plt.close('all')
plt.style.use(dn.styles['light_style_multi'])

def print_best(optrs):
    print('+++++++++++++++++++++++++++++++++++++++')
    best_params, best_posterior, best_run, _, _ = dn.get_best_optresult(optrs)
    
    print(f'Posterior: {best_posterior}')
    r = ''
    for k, v in best_params.to_dict('list').items():
        
        if v[0] == 0:
            ft = '{:.4f}'
            
        elif v[0] < 0.01 or v[0] > 1000:
            ft = '{:.4e}' 
            
        else:
            ft = '{:.4f}' 
        
        n = len(v)
        if n == 1:
            s = f'[{ft.format(v[0])}]*{n}'
        elif v[0] == v[1]:
            s = f'[{ft.format(v[0])}]*{n}'
        else:
            s = ', '.join([ft.format(v_) for v_ in v])
            s = f'[{s}]'
        
        spaces = ' '*(12 - len(k))
        s = f'{k}{spaces}: {s}'
        print(s)
        r = r + '\n' + s
    print('+++++++++++++++++++++++++++++++++++++++')
    return best_params, best_posterior, best_run, r

def get_dataset(model, filename='curvefit_04Glu02CA.csv', truncate=700):
    raw_data = pd.read_csv(filename, header=[0, 1], index_col=0)
    raw_data = raw_data.rename(columns=lambda x: int(float(x)), level=1).loc[0:truncate:3]
    
    
    data_dct = {k: v.values for k, v in raw_data.items()}
    time     = raw_data.index.values
    dataset  = {('Data', 0, 'x'): data_dct[('x', 0)],
                ('Time', 0, 'x'): time,
                ('Data', 1, 'x'): data_dct[('x', 1)],
                ('Time', 1, 'x'): time,
                ('Data', 0, 'R_frac'): data_dct[('R', 0)],
                ('Time', 0, 'R_frac'): time,
                ('Data', 1, 'R_frac'): data_dct[('R', 1)],
                ('Time', 1, 'R_frac'): time,
                ('Data', 0, 'H_frac'): data_dct[('H', 0)],
                ('Time', 0, 'H_frac'): time,
                ('Data', 1, 'H_frac'): data_dct[('H', 1)],
                ('Time', 1, 'H_frac'): time,
                }
    x     = raw_data.loc[:,('x', slice(None))]
    dx    = x.iloc[-1] - x.iloc[0]
    Yield = dx/1000*(1e6/110)
    Yield = Yield.values
    print('Yield', Yield)
    
    params = model.params
    params['Yield'] = Yield
    model.params = params
    
    init = model.states
    init['R'] = dataset[('Data', 0, 'R_frac')][0], dataset[('Data', 1, 'R_frac')][0]
    init['x'] = dataset[('Data', 0, 'x')][0], dataset[('Data', 1, 'x')][0]
    init['H'] = dataset[('Data', 0, 'H_frac')][0], dataset[('Data', 1, 'H_frac')][0]
    model.states = init
    
    temp         = {('Data', 0, 'mu'): data_dct[('mu', 0)],
                          ('Time', 0, 'mu'): time,
                          ('Data', 1, 'mu'): data_dct[('mu', 1)],
                          ('Time', 1, 'mu'): time,
                          ('Data', 0, 'syn_H'): data_dct[('syn_H', 0)],
                          ('Time', 0, 'syn_H'): time,
                          ('Data', 1, 'syn_H'): data_dct[('syn_H', 1)],
                          ('Time', 1, 'syn_H'): time,
                          }
    dataset_plot = {**dataset, **temp}
                                               
    
    return dataset, dataset_plot

def make_prez_AX(model, optrs, dataset, model_filename, data_filename, save=True, path=''):
    
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
    
    if optrs is None:
        dn.simulate_and_plot(model, AX, None, dataset, guess_marker='-')
        
    
    else:
        optrs = [o for o in optrs if hasattr(o, 'a')]
        best_params, best_posterior, best_run, string = print_best(optrs)
        
        dn.simulate_and_plot(model, AX, [optrs[best_run]], dataset, guess_marker='')
    
    AX['x'].set_title('(A) x', pad=10)
    AX['R_frac'].set_title('(B) ϕR', pad=10)
    AX['H_frac'].set_title('(C) ϕH', pad=10)
    AX[('mu', 'R_frac')].set_title('(D) ϕR vs λ', pad=10)
    AX_[0].legend(loc='upper left')
    
    if save:
        if path:
            fig.savefig(path)
        else:
            data_name  = data_filename.split('.')[0].split('_', 1)[1]
            model_name = model_filename.split('.')[0].split('_', 1)[1]
            new_filename = 'Model_' + model_name + '_' + data_name + '.png'
            fig.savefig(new_filename)
        

    return fig, AX

def make_AX(model):
    fig, AX_ = dn.figure(3, 6)

    AX                 = dict(zip(model.get_state_names(), AX_))
    AX['R_frac']       = AX.pop('R')
    AX['H_frac']       = AX.pop('H')
    AX['jH']           = AX_[-9]
    # AX['jH_']         = AX_[-8]
    AX['allprot']      = AX_[-7]
    AX['regR']         = AX_[-6]
    AX['jR']           = AX_[-5]
    AX['syn_R']        = AX_[-4]
    # AX['diff'] = AX_[-4]
    AX['A']            = AX_[-3]   
    AX[('mu', 'R_frac')]    = AX_[-2]
    AX[('mu', 'syn_H')] = AX_[-1]
    cc.plot_R_vs_mu(AX_[-2])
    cc.plot_synH_vs_mu(AX_[-1])
    AX[('mu', 'R_frac')].set_xlim(0, 0.02)
    AX[('mu', 'R_frac')].set_ylim(0, 0.3)
    AX[('mu', 'syn_H')].set_xlim(0, 0.02)
    AX[('mu', 'syn_H')].set_ylim(0, 5e-3)
    
    return fig, AX_, AX


    
if __name__ == '__main__':
    optimize = 0
    runs     = 50
    optrs    = []
    
    model_filename   = 'curvefit_G_6.dun'
    data_filename    = 'curvefit_04Glu.csv'
    dun_data, models = dn.read_file(model_filename)
    model            = models['M1']
    print(model_filename, data_filename)
    
    dataset, dataset_plot = get_dataset(model, data_filename)
    
    if optimize:
        # dn.fit_model(model, dataset, algo='simulated_annealing', n=runs, cache=optrs, new_seed=False)
        for i in range(runs):
            try:
                dn.fit_model(model, dataset, algo='simulated_annealing', n=1,   cache=optrs, new_seed=True)
            except:
                pass
    if optrs:
        for optr in optrs:
            pass
            # fig, AX_, AX = make_AX(model)
            
            # dn.simulate_and_plot(model, AX, [optr], dataset)
            # AX_[0].legend(loc='upper left')
    else:
        fig, AX_, AX = make_AX(model)
        dn.simulate_and_plot(model, AX, optrs, dataset_plot, guess_marker='-')
        AX_[0].legend(loc='upper left')
    
    if optimize:
        fig, AX_ = dn.figure(6, 4)
        
        for optr in optrs:
            for ax, var in zip(AX_, optr.free_params):
                try:
                    optr.plot_trace(var, ax)
                except:
                    pass
        
        make_prez_AX(model, optrs, dataset_plot, model_filename, data_filename, save=True)
        