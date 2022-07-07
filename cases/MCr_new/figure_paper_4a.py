import matplotlib
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
from scipy.stats         import ttest_ind


import addpath
import dunlin            as dn
import cell_calculation  as cc
import preprocess        as pp
from figure_paper_3 import get_new_params, adjust_yield

plt.close('all')
plt.ion()
plt.style.use('styles/paper_style_multi.mplstyle')

def plot_exp_OD(AX, skip0, skip1, trd0, trd1):   
    linestyle = '' 
    marker    = 'o'
    thin      = pp.thin
    label     = lambda scenario, ref, variable: f'{int(scenario[-1])}mM'
    
    for trd, ax1, skip in zip([trd0, trd1], AX, [skip0, skip1]):
        plot_args1 = {'marker'   : marker,   
                      'linestyle': linestyle, 
                      'color'    : pp.ind_color, 
                      'thin'     : thin,     
                      'skip'     : skip,
                      'xlabel'   : 'time (min)',
                      'ylabel'   : '$OD_{600}$',
                      'label'    : label,
                      'title'    : '',
                      **pp.errorcap,
                      }

        trd.plot_line(ax1, 'OD', **plot_args1)
        
        ax1.set_ylabel('$OD_{600}$')
        ax1.set_xlim(0,  750)
        ax1.set_ylim(0.0, 0.5)
    
    return

def plot_exp_R(AX, skip0, skip1, trd0, trd1):   
    linestyle = '' 
    marker    = 'o'
    thin      = pp.thin
    label     = lambda scenario, ref, variable: f'{scenario[-1]}mM IPTG'
    
    for trd, ax1, skip in zip([trd0, trd1], AX, [skip0, skip1]):
        plot_args1 = {'marker'   : marker,   
                      'linestyle': linestyle, 
                      'color'    : pp.ind_color, 
                      'thin'     : thin,     
                      'skip'     : skip,
                      'xlabel'   : 'time (min)',
                      'ylabel'   : '$ϕ_R$',
                      'label'    : label,
                      'title'    : '',
                      **pp.errorcap,
                      }

        trd.plot_line(ax1, 'R', **plot_args1)
        
        ax1.set_ylabel('$ϕ_R$')
        ax1.set_xlim(0,  750)
        ax1.set_ylim(0.0, 1.5)
    
    return

def setup_model(model, medium, data_filename, model_filename):
    mapping = {(medium, 0) : 0, (medium, 1) : 1}
    dataset = pp.trd1.reindex(['x', 'R', 'H', 'R_frac', 'H_frac', 'mu'], 
                              mapping, 
                              model=model,
                              no_fit={'R_frac', 'H_frac', 'mu'}
                              )
    new_params   = get_new_params(data_filename, model_filename)
    new_params   = pd.DataFrame(new_params)
    
    mul = 5
    print('Multiplying fH_var by 5')
    
    new_params['fH_var'] = new_params['fH_var']*mul 
    print(new_params['fH_var'])
    
    model.parameters = new_params
    dataset.adjust_model_init(model, ['R', 'H', 'x'])
    adjust_yield(dataset, model)

def plot_model_OD(AX, model_filename, 
           data_filename0, data_filename1, 
           medium0, medium1, 
           skip0, skip1):
    
    loaded = dn.load_file(model_filename)
    model  = loaded.parsed['Resource']
    zipped = zip(AX, [data_filename0, data_filename1], [medium0, medium1])
    
    for i, (ax1, data_filename, medium) in enumerate(zipped):
        #Set up model
        setup_model(model, medium, data_filename, model_filename)
        
        #Simulate
        sr = model.simulate()
        
        #Plot
        plot_args1 = {'color'    : pp.ind_colors,
                      'xlabel'   : 'time (min)',
                      'ylabel'   : '$OD_{600}$',
                      'title'    : '',
                      }

        sr.plot_line(ax1, 'x', **plot_args1)
        
        ax1.set_ylabel('$OD_{600}$')
        ax1.set_xlim(0,  750)
        ax1.set_ylim(0.0, 1.5)
        
    return model

def plot_model_R(AX, model_filename, 
           data_filename0, data_filename1, 
           medium0, medium1, 
           skip0, skip1):
    
    loaded = dn.load_file(model_filename)
    model  = loaded.parsed['Resource']
    zipped = zip(AX, [data_filename0, data_filename1], [medium0, medium1])
    
    for i, (ax1, data_filename, medium) in enumerate(zipped):
        #Set up model
        setup_model(model, medium, data_filename, model_filename)
        
        #Simulate
        sr = model.simulate()
        
        #Plot
        plot_args1 = {'color'    : pp.ind_colors,
                      'xlabel'   : 'time (min)',
                      'ylabel'   : '$ϕ_R$',
                      'title'    : '',
                      }

        sr.plot_line(ax1, 'R', **plot_args1)
        
        ax1.set_ylabel('$ϕ_R$')
        ax1.set_xlim(0,  750)
        ax1.set_ylim(0.0, 0.5)
        
    return model

def make_prez_AX(AX_, model, model_filename):
    
    AX = {}
    AX['x']      = AX_[0]
    AX['R_frac'] = AX_[1]
    AX['H_frac'] = AX_[2]
    AX[('mu', 'R_frac')]     = AX_[3]
    cc.plot_R_vs_mu(AX_[3])
    AX[('mu', 'R_frac')].set_xlim(0, 0.025)
    AX[('mu', 'R_frac')].set_ylim(0, 0.3)
    
    dn.simulate_and_plot(model, AX, None, None, guess_marker='-')
    
    AX_[0].set_title('x', pad=10, fontsize=12)
    AX_[1].set_title('$ϕ_R$', pad=10, fontsize=12)
    AX_[2].set_title('$ϕ_H$', pad=10, fontsize=12)
    AX_[3].set_title('$ϕ_R$ vs λ', pad=10, fontsize=12)
    
    legend_vals = model.params['fH_var'].values
    legend_vals = ['{:.2f}'.format(x) for x in legend_vals]
    
    AX_[0].legend(legend_vals, loc='lower right', title='fH_var')
    
    for ax in AX_:
        ax.set_title('')
        
    return fig, AX

def permute_param(AX, model_filename, data_filename, param_name='fH_var', 
                  span=np.logspace(-1, 2, 31), log=True,
                  axlabel='f_{H, var}'
                  ):
    '''exv: {'x': param_name, 'y': max_mu', 'z': final H_frac}
    '''
    loaded = dn.load_file(model_filename)
    model  = loaded.parsed['Resource']
    
    old_states = model.states
    
    old_params   = get_new_params(data_filename, model_filename)
    old_params   = pd.DataFrame(old_params)
    
    
    
    new_states = pd.concat([old_states.iloc[[0]]]*len(span), ignore_index=True)
    new_params = pd.concat([old_params.iloc[[0]]]*len(span), ignore_index=True)
    
    new_params[param_name] = span
    new_params['ind']      = 1
    
    model.states     = new_states
    model.parameters = new_params
    
    sr = dn.simulate_model(model)
    r  = sr[['final_H', 'max_mu']]
    r  = pd.DataFrame(r)
    
    r.index      = span
    r.index.name = param_name

    plot_args = dict(marker='', color=dn.get_color('cobalt'))
    AX[0].plot(r.index, r['max_mu'], **plot_args)
    AX[1].plot(r.index, r['final_H'], **plot_args)
    
    if log:
        AX[0].set_xscale('log')
        AX[1].set_xscale('log')
        AX[0].set_title(f'Effect on max λ', pad=5)
        AX[1].set_title(f'Effect on final $ϕ_H$', pad=5)
        AX[0].set_xlabel(f'log ${axlabel}$')
        AX[0].set_ylabel('max λ')
        AX[1].set_xlabel(f'log ${axlabel}$')
        AX[1].set_ylabel('max $ϕ_H$')
    else:
        AX[0].set_title(f'max lambda vs {param_name}', pad=10)
        AX[1].set_title(f'final $ϕ_H$ vs {param_name}', pad=10)
    
    AX[1].set_ylim(0, 0.5)
    
    return fig, AX

def ttest(medium, trd):
    R0 = trd.get('R', (medium, 0)).groupby(level=1).last()
    R1 = trd.get('R', (medium, 1)).groupby(level=1).last()
    stat, p = ttest_ind(R0.values, R1.values, )
    print('p value', medium, '{:.3e}'.format(p))
    # print(R0)
    # print(R1)
    
if __name__ == '__main__':
    matplotlib.rc('legend', fontsize=10)
    matplotlib.rc('xtick', labelsize=12)
    matplotlib.rc('ytick', labelsize=12)

    layout = []
    for ii in [0, 5]:
        for i in [2, 15]:
            layout.append([ii, ii+4, i, i+10])
    
    
    for i in [2, 15]:
        layout.append([12, 15, i, i+10])

    title   = ''
    fig, AX = dn.gridspec(18, 2, 
                          [[0, 5, 0, 1],
                           [0, 5, 1, 2],
                           [7, 12, 0, 1],
                           [7, 12, 1, 2],
                           [15, 18, 0, 1],
                           [15, 18, 1, 2],
                           ],
                          figsize=(8, 7), 
                          top=0.945, bottom=0.083, 
                          left=0.082, right=0.995,
                          wspace=0.4, hspace=1,
                          title=title
                          )
    
    ###########################################################################
    #Plot exp data
    trd0  = pp.trd1
    trd1  = pp.trd1
    skip0 = pp.g4
    skip1 = pp.base
    
    plot_exp_OD(AX[0:2], skip0, skip1, trd0, trd1)
    AX[0].legend(loc='upper left', title='IPTG')
    
    #Plot model predictions
    model_filename   = 'curvefit_G6.dunl'
    data_filename0   = 'curvefit_04Glu.csv'
    data_filename1   = 'curvefit_04Glu02CA.csv'
    medium0          = '0.4Glu'
    medium1          = '0.4Glu+0.2CA'
    
    model = plot_model_OD(AX[0:2], model_filename, 
                          data_filename0, data_filename1, 
                          medium0, medium1, 
                          skip0, skip1
                          )
    
    
    ###########################################################################
    #Plot exp data
    trd0  = pp.trd1
    trd1  = pp.trd1
    skip0 = pp.g4
    skip1 = pp.base
    
    plot_exp_R(AX[2:4], skip0, skip1, trd0, trd1)
    
    #Stat analysis
    ttest('0.4Glu+0.2CA', pp.trd1)
    ttest('0.4Glu', pp.trd1)
    print()
    
    #Plot model predictions
    model_filename   = 'curvefit_G6.dunl'
    data_filename0   = 'curvefit_04Glu.csv'
    data_filename1   = 'curvefit_04Glu02CA.csv'
    medium0          = '0.4Glu'
    medium1          = '0.4Glu+0.2CA'
    
    model = plot_model_R(AX[2:4], model_filename, 
                       data_filename0, data_filename1, 
                       medium0, medium1, 
                       skip0, skip1
                       )
    AX[0].set_title('0.4% Glu')
    AX[1].set_title('0.4% Glu + 0.2% CA')
    
    ###########################################################################
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)
    permute_param(AX[-2:], model_filename, data_filename0, param_name='fH_var', 
                  span=np.logspace(-1, 3, 41), log=True
                  )
    AX[-2].set_xticks([0.1, 1, 10, 100, 1000])
    AX[-1].set_xticks([0.1, 1, 10, 100, 1000])
    
    for i, ax in zip('ABCDEF', AX):
        ax.text(-0.2, 1.1, i, size=20, transform=ax.transAxes, fontweight='bold')
    
    fig.savefig('figures/4a.png', dpi=1200)