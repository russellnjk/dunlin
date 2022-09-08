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

def get_new_params(data_filename, model_filename):
    with open('cf_results.yaml', 'r') as file:
        yaml_data = ya.load(file, Loader=ya.FullLoader)
    
    yaml_data   = yaml_data[data_filename]['curvefit_G_6.dun']
    dunl_string = yaml_data['string']
    reader      = dn.standardfile.dunl.readstring.read_string
    new_params  = {}
    
    [new_params.update(reader(line)) for line in dunl_string.split('\n') if line]
    
    return new_params

def make_prez_AX(AX_, model, dataset, model_filename, data_filename):
        
    AX = {}
    AX['P']    = AX_[0]
    AX['rsat'] = AX_[1]
    
    
    sr = dn.simulate_model(model)
    dn.plot_line(AX, sr, ylabel='', xlabel='')
    
    for ax in AX_:
        ax.set_title('')
    
    AX['rsat'].set_ylim(0, 0.8)
    AX['P'].set_ylim(0, 1.4)
    
    return 

def protocol0(model_filename, data_filename):
    loaded = dn.load_file(model_filename)
    model  = loaded.parsed['Resource']
    print(model_filename, data_filename)
    
    medium  = '0.4Glu'
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
    matplotlib.rc('axes', labelsize=14, titlesize=14, titlepad=20)
    matplotlib.rc('xtick', labelsize=12)
    matplotlib.rc('ytick', labelsize=12)
    
    layout = []
    for i in range(0, 2):
        for ii in range(2):
            layout.append([i, i+1, ii, ii+1])
    
    title   = ''
    fig, AX = dn.gridspec(2, 2, 
                          layout,
                          figsize=(8, 5), 
                          top=0.91, bottom=0.15, 
                          left=0.08, right=0.99,
                          wspace=0.3, hspace=0.5,
                          title=title
                          )
    
    model_filename   = 'curvefit_G6.dunl'
    data_filename    = 'curvefit_04Glu.csv'
    model, dataset_plot = protocol0(model_filename, data_filename)
    make_prez_AX(AX[0::2], model, dataset_plot, model_filename, data_filename)
    
    model_filename   = 'curvefit_G6.dunl'
    data_filename    = 'curvefit_04Glu02CA.csv'
    model, dataset_plot = protocol0(model_filename, data_filename)
    make_prez_AX(AX[1::2], model, dataset_plot, model_filename, data_filename)
    
    AX[0].set_title('0.4% Glu', pad=20)
    AX[1].set_title('0.4% Glu + 0.2% CA', pad=20)
    AX[2].set_xlabel('time (min)')
    AX[3].set_xlabel('time (min)')
    AX[0].set_ylabel('P')
    AX[2].set_ylabel('RA')
    AX[0].legend(title='IPTG')
    
    for i, ax in zip('ABCDEF', AX):
        ax.text(-0.2, 1.1, i, size=20, transform=ax.transAxes, fontweight='bold')
    
    AX[0].text(0, -1.98, 'Note: Concentrations in arbitrary units.', size=12, transform=AX[0].transAxes)
    
    fig.savefig('figures/RA.png', dpi=1200)