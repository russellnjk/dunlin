import matplotlib
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import yaml              as ya

import addpath
import dunlin as dn
import cell_calculation as cc
import preprocess       as pp
from fitmodel import get_from_dunl, adjust_yield

plt.close('all')
plt.ion()
plt.style.use('styles/paper_style_multi.mplstyle')

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

def get_new_params(data_filename, model_filename):
    with open('cf_results.yaml', 'r') as file:
        yaml_data = ya.load(file, Loader=ya.FullLoader)
    
    yaml_data   = yaml_data[data_filename]['curvefit_G_6.dun']
    dunl_string = yaml_data['string']
    reader      = dn.standardfile.dunl.readstring.read_string
    new_params  = {}
    
    [new_params.update(reader(line)) for line in dunl_string.split('\n') if line]
    
    return new_params

def make_prez_AX(AX_, model, dataset, model_filename, data_filename, thin=5):
        
    AX = {}
    AX['x']      = AX_[0]
    AX['R_frac'] = AX_[1]
    AX['H_frac'] = AX_[2]
    AX[('mu', 'R_frac')]     = AX_[3]
    cc.plot_R_vs_mu(AX_[3])
    
    AX['x'].set_ylim(0, 1.5)
    AX['R_frac'].set_ylim(0, 0.3)
    AX['H_frac'].set_ylim(0, 0.3)
    
    AX[('mu', 'R_frac')].set_xlim(0, 0.02)
    AX[('mu', 'R_frac')].set_ylim(0, 0.3)
    
    
    thin = {'x': 5, 'R_frac': 5, 'H_frac': 5, ('mu', 'R_frac'): thin}
    sr = dn.simulate_model(model)
    dn.plot_line(AX, sr, ylabel='', xlabel='')
    
    for key in AX:
        
        dn.plot_line({key: AX[key]}, dataset, label='_nolabel', thin=thin[key], ylabel='', xlabel='')
    
    for ax in AX_:
        ax.set_title('')
        
    return 

def protocol0(model_filename, data_filename, medium):
    loaded = dn.load_file(model_filename)
    model  = loaded.parsed['Resource']
    print(model_filename, data_filename)
    
    # medium  = '0.4Glu'
    mapping = {(medium, 0) : 0, (medium, 1) : 1}
    dataset = pp.trd0.reindex(['x', 'R', 'H', 'R_frac', 'H_frac', 'mu'], 
                              mapping, 
                              model=model,
                              no_fit={'R_frac', 'H_frac', 'mu'}
                              )
    
    new_params = get_new_params(data_filename, model_filename)
    
    model.parameters = new_params
    dataset.adjust_model_init(model, ['R', 'H', 'x'])
    adjust_yield(dataset, model)
    
    return model, dataset
    

if __name__ == '__main__':
    matplotlib.rc('legend', fontsize=8, title_fontsize=8, handlelength=1.2)
    matplotlib.rc('axes', labelsize= 10, titlesize=12, titlepad=0)
    matplotlib.rc('xtick', labelsize=12)
    matplotlib.rc('ytick', labelsize=12)


    path = 'figures/CF.png'
    save = True 
    
    
    model_pic = [0, 2, 0, 4]
    layout = [model_pic]
    for i in range(2, 5):
        for ii in range(4):
            layout.append([i, i+1, ii, ii+1])
    
    title   = 'Model Development'
    title   = ''
    fig, AX = dn.gridspec(5, 4, 
                          layout,
                          figsize=(8, 9.5), 
                          top=0.99, bottom=0.055, 
                          left=0.05, right=0.99,
                          wspace=0.4, hspace=0.8,
                          title=title
                          )
    
    img = mpimg.imread('figures/Model G2.png')
    # pad = 15
    # img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)))
    AX[0].imshow(img)
    AX[0].axes.xaxis.set_ticklabels([])
    AX[0].axes.yaxis.set_ticklabels([])
    AX[0].set_xticks([])
    AX[0].set_yticks([])
    for spine in AX[0].spines.values():
        spine.set_color('white')
    
    
    AX_ = AX[1:]
    model_filename   = 'curvefit_G6.dunl'
    data_filename    = 'curvefit_04Glu.csv'
    model, dataset_plot = protocol0(model_filename, data_filename, '0.4Glu')
    make_prez_AX(AX_[0:4], model, dataset_plot, model_filename, data_filename)
    
    model_filename   = 'curvefit_G6.dunl'
    data_filename    = 'curvefit_04Gly.csv'
    model, dataset_plot = protocol0(model_filename, data_filename,  '0.4Gly')
    make_prez_AX(AX_[4:8], model, dataset_plot, model_filename, data_filename)
    
    model_filename   = 'curvefit_G6.dunl'
    data_filename    = 'curvefit_04Glu02CA.csv'
    model, dataset_plot = protocol0(model_filename, data_filename,  '0.4Glu+0.2CA')
    make_prez_AX(AX_[8:12], model, dataset_plot, model_filename, data_filename, thin=1)
    
    AX_[0].set_title('x', pad=10)
    AX_[1].set_title('$ϕ_R$', pad=10)
    AX_[2].set_title('$ϕ_H$', pad=10)
    AX_[3].set_title('$ϕ_R$ vs λ', pad=10)
    AX_[0].legend(loc='lower right', title='IPTG')
    
    AX[0].text(0, 0.9, 'A', size=20, transform=AX[0].transAxes, fontweight='bold')
    
    i = 0
    for ax in AX[1:]:
        i += 1
        ax.text(0.05, 1.1, i, size=18, transform=ax.transAxes, fontweight='bold')
        
        if i < 4:
            ax.set_xlabel('time (min)')
        else:
            ax.set_xlabel('λ (1/min)    ')
        
        if i == 4:
            i = 0
    
    for i, ax in zip('BCDE', AX[1::4]):
        ax.text(-0.28, 1.1, i, size=20, transform=ax.transAxes, fontweight='bold')
    
    AX[1].text(0, 1.4, 'Note: Time is in minutes, λ is in 1/minute and IPTG conc. is in mM', size=11, transform=AX[1].transAxes)
    
    fig.savefig('figures/3.png', dpi=1200)