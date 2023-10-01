import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
from pathlib import Path

import addpath
import dunlin            as dn
import cell_calculation  as cc

def GFP2H(ref):
    def helper(gfp, od):
        stoich   = 5.3  if 'LZ' in ref else 1
        medium   = 'M9'
        
        new_series = cc.gfp_od2hfrac(gfp, od, stoich=stoich, medium=medium)  
        
        return new_series
    return helper

def muR2A(mu, R):
    return mu/R/0.211
    
###############################################################################
#Script
###############################################################################
loaded = dn.load_file('ResourceData.dunl')
trd0   = loaded.parsed['GFP']
trd1   = loaded.parsed['LZGFP'] 

for trd in [trd0, trd1]:
    trd.dup('OD600', 'OD')
    trd.spec_diff('OD', name='mu')
    trd.apply(cc.rfp_od2rfrac, 'RFP', 'OD', name='R')
    trd.first_order_gen('R', 'mu', name='synR')
    trd.apply(GFP2H(trd.ref), 'GFP', 'OD', name='H')
    trd.apply(muR2A, 'mu', 'R', name='A')#Max transl rate = 0.211 /min #Alternatively 0.1769
    trd.dup('OD', name='x', no_fit=False)
    trd.dup('R', name='R_frac', no_fit=False)
    trd.dup('H', name='H_frac', no_fit=False)
    
###############################################################################
#Utils for Downstream
###############################################################################
def select(medium, ind=(0, 1)):
    def skip(scenario):
        if scenario[0] == medium:
            if scenario[1] == ind or scenario[1] in ind:
                return False
        
        return True
    return skip


ind_colors = {0: 'cobalt', 0.1: 'ocean', 1: 'coral'}
ind_color  = lambda scenario, variable, ref: ind_colors[scenario[-1]]
linestyles = {0: '-', 0.1: ':', 1: '--'}
linestyle  = lambda scenario, variable, ref: linestyles[scenario[-1]]
errorcap   = dict(capsize=5, elinewidth=1.5, capthick=1)
thin       = 2

#Scenario selectors
base = select('0.4Glu+0.2CA')
g4c2 = base
g4   = select('0.4Glu')
y4   = select('0.4Gly')
y2   = select('0.2Gly')

glu = lambda scenario: 'Glu' in scenario[0] and scenario[1] == 0
gly = lambda scenario: 'Gly' in scenario[0] and scenario[1] == 0

if __name__ == '__main__':
    plt.close('all')
    plt.ion()
    
    skip = select('0.4Glu')
    
    fig, AX_ = dn.figure(4, 3)
    to_plot  = ['OD', 'x', 'mu', 'R', 'H', ('mu', 'R')]
    AX       = dict(zip(to_plot, AX_))
    
    line_args = dict(skip=g4, 
                     color=ind_color, 
                     thin=thin,
                     **errorcap
                     )
    
    
    AX_[0].set_title('OD$_\mathregular{{600}}$')

    dn.plot_line(AX, trd0, **line_args)
    
    trd0.plot_line(AX_[-1], 'OD', title='OD$_\mathregular{{600}}$', **line_args)
    
    