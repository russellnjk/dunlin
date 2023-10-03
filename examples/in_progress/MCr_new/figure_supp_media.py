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

def plot_media(ax, title, scenario, color, marker):
    skip    = lambda c: c != scenario 
    label   = lambda ref, variable, scenario: scenario[0]
    
    plot_args = {'marker'   : marker, 'linestyle': '-',  'color'     : color, 
                 'thin'     : pp.thin, 'label'    : '_nolabel',
                 'xlabel': 'λ (1/min)',
                 'ylabel': '$ϕ_R$',
                 'variable' : ('mu', 'R'),
                 'alpha'    : 0.3,
                 **pp.errorcap
                 }
    
    pp.trd0.plot_line(ax, title=title, skip=skip, **plot_args)
    
    plot_args = {'marker'   : marker, 'linestyle': '',  'color'     : color, 
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
    fig, AX    = dn.figure(3, 2, 
                           figsize=(8, 9.2),
                           top=0.952, bottom=0.06, left=0.1, right=0.975,
                           hspace=0.75, wspace=0.4,
                           title=title
                           )
    for i, ax in zip('ABCDEFG', AX):
        ax.text(-0.2, 1.1, i, size=20, transform=ax.transAxes, fontweight='bold')
    
    ###########################################################################
    colors = {'0.4Glu'       : 'dark teal', 
              '0.4Gly'       : 'dark teal',
              '0.4Glu+0.2CA' : 'crimson',
              '0.4Gly+0.2CA' : 'crimson',
              '0.8Glu+0.4CA' : 'cobalt',
              '0.2Gly'       : 'dark orange',
              }
    
    colors = {(k, 0): v for k, v in colors.items()}
    
    markers = {'0.4Glu'       : 'o', 
               '0.4Glu+0.2CA' : '^',
               '0.8Glu+0.4CA' : 's',
               '0.2Gly'       : 's',
               '0.4Gly'       : 'o',
               '0.4Gly+0.2CA' : '^'
               }
    
    
    for ax, (scenario, color) in zip(AX, colors.items()):
        medium = scenario[0]
        marker = markers[medium] 
        title  = medium
        plot_media(ax, title, scenario, color, marker)
    
    
    ###########################################################################
    fig.savefig('figures/supp_media.png', dpi=1200)