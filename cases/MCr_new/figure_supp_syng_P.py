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
    
    AX['x'].set_xlabel('time (min)')
    AX['P'].set_xlabel('time (min)')
    AX[('P', 'syng_eff')].set_xlabel('P')
    AX[('P', 'jR')].set_xlabel('P')
    
    AX['x'].set_ylabel('x')
    AX['P'].set_ylabel('P')
    AX[('P', 'syng_eff')].set_ylabel('$syn_g$')
    AX[('P', 'jR')].set_ylabel('$J_R$')
    
    #twinx    
    AX['Rel']              = twin = AX['P'].twinx()
    line_args              = sr.sim_args.get('line_args', {})
    line_args['linestyle'] = '--'
    dn.plot_line({'Rel': twin}, 
                 sr, 
                 ylabel=None, 
                 xlabel=None, 
                 skip=skip, 
                 **line_args
                 )
    twin.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3))
    twin.set_ylabel('RelA')
    
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
    
    layout = []
    for i in range(0, 2):
        for ii in range(2):
            layout.append([i, i+1, ii, ii+1])
    
    title   = ''
    fig, AX = dn.gridspec(2, 2, 
                          layout,
                          figsize=(8, 5), 
                          top=0.926, bottom=0.140, 
                          left=0.076, right=0.942,
                          wspace=0.3,  hspace=0.8,
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
    #syng vs P
    df = get_at_time(sr, ('P', 'syng_eff'))
    
    AX[2].annotate('t=60', 
                   xy=df.loc[60],
                   xytext=[15, 10],
                   **ann_args
                   )
    AX[2].annotate('t=120', 
                   xy=df.loc[120],
                   xytext=[35, 15],
                   **ann_args
                   )
    AX[2].annotate('t=240', 
                   xy=df.loc[240],
                   xytext=[-25, 15],
                   **ann_args
                   )
    AX[2].annotate('t=360', 
                   xy=df.loc[360],
                   xytext=[-5, 25],
                   **ann_args
                   )
    AX[2].annotate('t=540', 
                   xy=df.loc[540],
                   xytext=[10, 25],
                   **ann_args
                   )
    
    #jR vs P
    df = get_at_time(sr, ('P', 'jR'))
    
    AX[3].annotate('t=60', 
                   xy=df.loc[60],
                   xytext=[25, 15],
                   **ann_args
                   )
    AX[3].annotate('t=120', 
                   xy=df.loc[120],
                   xytext=[35, 5],
                   **ann_args
                   )
    AX[3].annotate('t=240', 
                   xy=df.loc[240],
                   xytext=[-25, 15],
                   **ann_args
                   )
    AX[3].annotate('t=360', 
                   xy=df.loc[360],
                   xytext=[-5, 15],
                   **ann_args
                   )
    AX[3].annotate('t=540', 
                   xy=df.loc[540],
                   xytext=[15, 15],
                   **ann_args
                   )
    
    ###########################################################################
    AX[0].legend(title='IPTG')
    AX[0].text(0, -2.28, 'Note: Units are arbitrary unless stated otherwise.', size=12, transform=AX[0].transAxes)
    
    for i, ax in zip('ABCD', AX):
        ax.text(-0.1, 1.1, i, size=20, transform=ax.transAxes, fontweight='bold')
    
    fig.savefig('figures/syng_P.png', dpi=1200)