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

def find_nearest_timepoints(series, timepoints=[60, 120, 240, 360, 540]):
    
    if series.index.nlevels > 1: 
        all_timepoints = series.index.levels[0]
    else:
        all_timepoints = series.index
    
    idxs               = all_timepoints.get_indexer(timepoints, 'nearest')
    nearest_timepoints = all_timepoints[idxs]
    
    return list(nearest_timepoints)

def get_at_time(dataset, state, scenario=0, timepoints=[60, 120, 240, 360, 540]):
    if type(state) == str:
        series             = dataset[state, scenario]
        nearest_timepoints = find_nearest_timepoints(series, timepoints)
        values             = series.loc[nearest_timepoints]
    else:
        dct = dataset[list(state), scenario]
        dct = {key: value[scenario] for key, value in dct.items()}
        df  = pd.DataFrame(dct)
        
        nearest_timepoints = find_nearest_timepoints(df, timepoints)
        values             = df.loc[nearest_timepoints]
    if values.index.nlevels > 1:
        values = values.groupby(axis=0, level=0).mean()
        
    return values

def annotate(dataset, ax, state, scenario=0):
    tps    = [60, 120, 240, 360, 540]
    values = get_at_time(dataset, 
                         state, 
                         scenario=0, 
                         timepoints=tps
                         )
    
    if values.index.nlevels > 1:
        values = values.groupby(axis=0, level=0).mean()
    
    arrowprops = dict(facecolor='black', 
                      lw=0.1
                      )
    
    if type(values) == pd.DataFrame:
        ax.annotate('t=60', 
                    xy=values.loc[60],
                    xycoords='data',
                    xytext=[5, -45],
                    textcoords='offset points',
                    arrowprops=arrowprops
                    )
        
        
        # for time, row in values.iterrows():
        #     ax.annotate(f't={time}', 
        #                 xy=row.values,
        #                 xycoords='data',
        #                 xytext=[0.1, 0.1],
        #                 textcoords='figure points'
        #                 )
            
        print(values)
    
def get_new_params(data_filename, model_filename):
    with open('cf_results.yaml', 'r') as file:
        yaml_data = ya.load(file, Loader=ya.FullLoader)
    
    yaml_data   = yaml_data[data_filename]['curvefit_G_6.dun']
    dunl_string = yaml_data['string']
    reader      = dn.standardfile.dunl.readstring.read_string
    new_params  = {}
    
    [new_params.update(reader(line)) for line in dunl_string.split('\n') if line]
    
    return new_params

def make_prez_AX(AX_, model, model_filename, data_filename):
        
    AX = {}
    AX['x']                 = AX_[0] 
    AX['P']                 = AX_[1]
    AX[('P', 'syng_eff')]   = AX_[2]
    AX[('P', 'jR')]         = AX_[3]
    AX[('mu', 'syng_eff')]  = AX_[4]
    AX[('mu', 'jR')]        = AX_[5]
    AX[('mu', 'R_frac')]    = AX_[6]

    skip= lambda c: c == 1
    sr = dn.simulate_model(model)
    dn.plot_line(AX, sr, ylabel=None, xlabel=None, skip=skip)
    
    for ax in AX_:
        ax.set_title('')
    
    pad = 5
    AX['x'].set_title('x', pad=pad)
    AX['P'].set_title('P', pad=pad)
    AX[('P', 'syng_eff')].set_title('$syn_g$ vs P', pad=pad)
    AX[('P', 'jR')].set_title('$J_R$ vs P', pad=pad)
    AX[('mu', 'syng_eff')].set_title('$syn_g$ vs λ', pad=pad)
    AX[('mu', 'jR')].set_title('$J_R$ vs λ', pad=pad)
    AX[('mu', 'R_frac')].set_title('$ϕ_R$ vs λ', pad=pad)
    
    AX['x'].set_xlabel('time (min)')
    AX['P'].set_xlabel('time (min)')
    AX[('P', 'syng_eff')].set_xlabel('P')
    AX[('P', 'jR')].set_xlabel('P')
    AX[('mu', 'syng_eff')].set_xlabel('λ (1/min)')
    AX[('mu', 'jR')].set_xlabel('λ (1/min)')
    AX[('mu', 'R_frac')].set_xlabel('λ (1/min)')
    
    AX['x'].set_ylabel('x')
    AX['P'].set_ylabel('P')
    AX[('P', 'syng_eff')].set_ylabel('$syn_g$')
    AX[('P', 'jR')].set_ylabel('$J_R$')
    AX[('mu', 'syng_eff')].set_ylabel('$syn_g$')
    AX[('mu', 'jR')].set_ylabel('$J_R$')
    AX[('mu', 'R_frac')].set_ylabel('$ϕ_R$')
    
    return sr

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
    # matplotlib.rc('axes', labelsize=14, titlesize=14, titlepad=20)
    matplotlib.rc('xtick', labelsize=12)
    matplotlib.rc('ytick', labelsize=12)


    path = 'figures/CF.png'
    save = True 
    
    # layout = []
    # for i in range(0, 3):
    #     for ii in range(2):
    #         layout.append([i, i+1, 2*ii, 2*(ii+1)])
    # layout.append([3, 5, 1, 3])
    
    # title   = ''
    # fig, AX = dn.gridspec(5, 4, 
    #                       layout,
    #                       figsize=(8, 8), 
    #                       top=0.892, bottom=0.064, 
    #                       left=0.11, right=0.99,
    #                       wspace=1,  hspace=1,
    #                       title=title
    #                       )
    layout = []
    for i in range(0, 3):
        for ii in range(2):
            layout.append([5*i, 5*(i+1)-2, 2*ii, 2*(ii+1)])
    layout.append([15, 19, 1, 3])
    
    title   = ''
    fig, AX = dn.gridspec(19, 4, 
                          layout,
                          figsize=(8, 8), 
                          top=0.972, bottom=0.098, 
                          left=0.11, right=0.99,
                          wspace=1,  hspace=1,
                          title=title
                          )
    
    model_filename = 'curvefit_G6.dunl'
    data_filename  = 'curvefit_04Glu.csv'
    model, dataset = protocol0(model_filename, data_filename)
    sr             = make_prez_AX(AX, model, model_filename, data_filename)
    
    ###########################################################################
    #Annotate
    ###########################################################################
    arrowprops = dict(arrowstyle='->'
                      )
    ann_args   = dict(xycoords='data', 
                      textcoords='offset points',
                      arrowprops=arrowprops
                      )
    df = get_at_time(sr, ('mu', 'R'))
    
    AX[-1].annotate('t=60', 
                    xy=df.loc[60],
                    xytext=[5, -35],
                    **ann_args
                    )
    AX[-1].annotate('t=120', 
                    xy=df.loc[120],
                    xytext=[5, -35],
                    **ann_args
                    )
    AX[-1].annotate('t=240', 
                    xy=df.loc[240],
                    xytext=[-55, 5],
                    **ann_args
                    )
    AX[-1].annotate('t=360', 
                    xy=df.loc[360],
                    xytext=[5, 15],
                    **ann_args
                    )
    AX[-1].annotate('t=540', 
                    xy=df.loc[540],
                    xytext=[5, 15],
                    **ann_args
                    )
    
    #syng_eff vs mu
    df = get_at_time(sr, ('mu', 'syng_eff'))
    
    AX[4].annotate('t=60', 
                   xy=df.loc[60],
                   xytext=[-5, 25],
                   **ann_args
                   )
    AX[4].annotate('t=120', 
                   xy=df.loc[120],
                   xytext=[-55, 5],
                   **ann_args
                   )
    AX[4].annotate('t=240', 
                   xy=df.loc[240],
                   xytext=[-25, 15],
                   **ann_args
                   )
    AX[4].annotate('t=360', 
                   xy=df.loc[360],
                   xytext=[5, 15],
                   **ann_args
                   )
    AX[4].annotate('t=540', 
                   xy=df.loc[540],
                   xytext=[-15, 45],
                   **ann_args
                   )
    
    ###########################################################################
    AX[0].legend(title='IPTG')
    AX[0].text(0, -7.2, 'Note: Units are arbitrary unless stated otherwise.', size=12, transform=AX[0].transAxes)
    
    fig.savefig('figures/syng.png', dpi=1200)