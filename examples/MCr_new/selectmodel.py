import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

import addpath
import dunlin as dn
import cell_calculation as cc
import preprocess       as pp 
import fitmodel         as fm

plt.close('all')
plt.ion()

all_data = dn.read_file('cf_result.dunl')

medium = '0.4Glu'

k  = medium.replace('.', '')
d  = {ref: {'objective' : dct[k]['objective'], 'n_free': dct[k]['n_free']} for ref, dct in all_data.items()}
df = pd.DataFrame(d).T

mapping = {(medium, 0) : 0, (medium, 1) : 1}
dataset = pp.trd0.reindex(['x', 'R', 'H', 'R_frac', 'H_frac', 'mu'], 
                          mapping, 
                          no_fit={'R_frac', 'H_frac', 'mu'}
                          )

df['n_points'] = dataset.n_points


objective = df['objective']
n_points  = df['n_points']
n_free    = df['n_free']

df['AIC'] = 2*n_free + n_points* np.log(objective/n_points)

print(df)