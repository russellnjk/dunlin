import matplotlib
import matplotlib.pyplot as plt
import numpy            as np
import pandas           as pd
import seaborn          as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib                               import Path
from scipy.stats                           import ttest_ind

plt.close('all')
plt.ion()
plt.style.use('styles/paper_style_multi.mplstyle')

import addpath
import dunlin                  as dn
import cell_calculation as cc
import preprocess       as pp
from figure_paper_4 import ttest

def compare_cycle(ax0, ax1, title, skip):
    label     = lambda ref, variable, scenario: f'{int(scenario[1])}mM '
    plot_args = dict(xlabel='time (min)', 
                     ylabel='λ',
                     marker= 'o', 
                     linewidth= 1.5, 
                     color=pp.ind_color, 
                     linestyle='None',
                     label=label, 
                     thin=pp.thin,
                     **pp.errorcap
                     )
    pp.trd0.plot_line(ax0, 'mu', title=title, skip=skip, **plot_args)
    ax0.set_xlim(0, 800)
    ax0.set_ylim(0, 0.02)
    ax0.set_title(title, pad=20)
    ax0.legend(title='IPTG')
    
    plot_args = {'linestyle': pp.linestyle,  
                 'color'    : pp.ind_color, 
                 'thin'     : pp.thin if 'Gly' in title else 1, #Needs more points
                 'label'    : 'None',
                 'xlabel'   : 'λ (1/min)',
                 'ylabel'   : '$ϕ_R$',
                 'variable' : ('mu', 'R'),
                 **pp.errorcap
                 }
    
    pp.trd0.plot_line(ax1, skip=skip, **plot_args)
    ax1.set_xlim(0, 0.02)
    ax1.set_ylim(0, 0.30)
    cc.plot_R_vs_mu(ax1)

def compare_media(ax, skip, title, colors, markers):
    
    label   = lambda ref, variable, scenario: scenario[0]
    plot_args = {'marker'   : markers, 'linestyle': '-',  'color'     : colors, 
                 'thin'     : pp.thin, 'label'    : label,
                 'xlabel': 'λ (1/min)',
                 'ylabel': '$ϕ_R$',
                 'variable' : ('mu', 'R'),
                 **pp.errorcap
                 }
    
    pp.trd0.plot_line(ax, title=title, skip=skip, **plot_args)
    ax.set_xlim(0, 0.02)
    ax.set_ylim(0, 0.30)
    
    # get handles
    handles, labels = ax.get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    # use them in the legend
    ax.legend(handles, labels, numpoints=1)

    # ax.legend()
    cc.plot_R_vs_mu(ax)
    
if __name__ == '__main__':
    matplotlib.rc('legend', fontsize=9)
    # matplotlib.rc('axes', labelsize= 14, titlesize=14, titlepad=10)
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)
    
    ###############################################################################
    #Changing Media and H Expression
    ###############################################################################
    title      = ''
    fig, AX    = dn.gridspec(23, 2,
                             [(0, 5, 0, 1),
                              (0, 5, 1, 2),
                              (8, 11, 0, 1),
                              (8, 11, 1, 2),
                              (12, 16, 0, 1),
                              (12, 16, 1, 2),
                              (18, 23, 0, 2),
                             ], 
                             figsize=(8, 9.2),
                             top=0.955, bottom=0.03, left=0.1, right=0.975,
                             hspace=1, wspace=0.4,
                             title=title
                             )
    for i, ax in zip('ABCD', AX):
        ax.text(-0.2, 1.1, i, size=20, transform=ax.transAxes, fontweight='bold')
    
    ax = AX[-1]
    ax.text(-0.05, 1.1, 'E', size=20, transform=ax.transAxes, fontweight='bold')
    
    ###########################################################################
    colors = {'0.4Glu'       : 'dark teal', 
              '0.4Glu+0.2CA' : 'crimson',
              '0.8Glu+0.4CA' : 'cobalt',
              '0.2Gly'       : 'dark orange',
              '0.4Gly'       : 'dark teal',
              '0.4Gly+0.2CA' : 'crimson'
              }
    
    colors = {(k, 0): v for k, v in colors.items()}
    
    markers = {'0.4Glu'       : 'o', 
                '0.4Glu+0.2CA' : '^',
                '0.8Glu+0.4CA' : 's',
                '0.2Gly'       : 's',
                '0.4Gly'       : 'o',
                '0.4Gly+0.2CA' : '^'
                }
    
    markers = {(k, 0): v for k, v in markers.items()}
    
    skip = lambda c: 'Glu' not in c[0] or c[1] != 0 or c not in colors
    ax   = AX[0]
    title = 'Glucose + CA'
    compare_media(ax, skip, title, colors, markers)
    
    skip = lambda c: 'Gly' not in c[0] or c[1] != 0 or c not in colors
    ax   = AX[1]
    title = 'Glycerol + CA'
    compare_media(ax, skip, title, colors, markers)
    
    ###########################################################################
    skip = pp.y2
    ax   = AX[2], AX[4]
    title = '0.2% Gly'
    compare_cycle(*ax, title, skip)
    
    skip = pp.g4c2
    ax   = AX[3], AX[5]
    title = '0.4% Glu+0.2%CA'
    compare_cycle(*ax, title, skip)
    
    ###########################################################################
    ax = AX[-1]
    
    final_R = {}
    cs = [('0.2Gly', 0), 
          ('0.2Gly', 1), 
          ('0.4Glu+0.2CA', 0), 
          ('0.4Glu+0.2CA', 1)
         ]
    R = pp.trd0['R', cs]['R']
    for c, series in R.items():
        final_R[c] = series.loc[729]
        
    final_R = pd.DataFrame(final_R)
    mean    = final_R.mean(axis=0).unstack()
    sd      = final_R.std(axis=0).unstack()
    
    color = [dn.get_color(pp.ind_colors[i]) for i in (0, 1)]
    mean.plot.bar(ax=ax, yerr=sd, color=color, capsize=5, rot=0, width=0.4)
    ax.legend(['0mM', '1mM'], title='IPTG')
    
    ttest('0.2Gly', pp.trd0)
    ttest('0.4Glu+0.2CA', pp.trd0)
    
    ###########################################################################
    fig.savefig('figures/2.png', dpi=1200)