import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

import addpath
import dunlin as dn
import cell_calculation  as cc

plt.ion()
plt.close('all')

def make_AX(model):
    fig, AX_ = dn.figure(3, 6)

    AX                 = dict(zip(model.state_names, AX_))
    AX['R_frac']       = AX.pop('R')
    AX['H_frac']       = AX.pop('H')
    AX['jH']           = AX_[-9]
    AX['allprot']      = AX_[-7]
    AX['regR']         = AX_[-6]
    AX['jR']           = AX_[-5]
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
    
    return fig, AX_, AX

model_filename = 'testResource.dunl'
data_filename  = 'testData.dunl'

loaded  = dn.load_file(model_filename, data_filename)
model   = loaded.parsed['Resource'] 
dataset = loaded.parsed['GFP']
dataset.dup('R', 'R_frac', no_fit=False)
dataset.dup('H', 'H_frac', no_fit=True)

runs = 1

if runs:
    curvefitters = dn.fit_model(model, dataset, runs, algo='simulated_annealing')
else:
    curvefitters = []


colors    = {0: 'cobalt', 1: 'crimson'}
line_args = {'color': colors}

fig, AX_, AX = make_AX(model)
dn.plot_curvefit(AX, curvefitters, dataset, model, plot_guess=True, 
                 guess_line_args=line_args, 
                 data_line_args=line_args, 
                 posterior_line_args=line_args
                 )
AX_[0].legend()

fig, AX = dn.figure(1, 1)
sim_result = dn.simulate_model(model)
sim_result.plot_bar(AX[0], 'final_H')
