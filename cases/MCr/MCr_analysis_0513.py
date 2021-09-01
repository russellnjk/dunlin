import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import matplotlib.ticker as mtick
from matplotlib.collections import LineCollection

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin               as dn
import dunlin.simulate    as sim
import dunlin.curvefit      as cf
import dunlin.dataparser    as dp
import dunlin.traceanalysis as ta

add   = lambda x, y: x + y
minus = lambda x, y: x - y
mul   = lambda x, y: x * y
div   = lambda x, y: x / y

def convert_rfp(au_od):
    au2M   = 4.35e9
    od2caa = 3.6e9*1e3*12e8
    au2gaa = 1/au2M*6e23*225
    return au_od*au2gaa/od2caa
convert_rfp(1800)

def convert_gfp(au_od):
    au2M   = 2.77e11
    od2caa = 3.6e9*1e3*12e8
    au2gaa = 1/au2M*6e23*236
    return au_od*au2gaa/od2caa
convert_gfp(2.5e6)

def apply2data(data, name, func, *states, **kwargs):
    if type(func) == str:
        if func == '+':
            func_ = add
        elif func == '-':
            func_ = minus
        elif func == '*':
            func_ = mul
        elif func == '/':
            func_ = div
    elif isnum(func):
        func_ = lambda x: x*func
    elif type(func) == np.ndarray:
        func_ = lambda x: x.multiply(func, axis=0)
    else:
        func_ = func
    
    vectors = [data[s] for s in states]
    new     = func_(*vectors, **kwargs)

    data[name] = new
    
    return data

def isnum(x):
    try:
        float(x)
        return True
    except:
        return False

def dai(mu):
    gradient  = 5.78656638987421
    intercept = 0.03648482880435973
    
    return mu*gradient + intercept

plt.close('all')
plt.style.use(dn.styles['dark_style_multi'])

raw  = pd.read_csv('data_MCr_20210513.csv', header=[0, 1, 2])
idx  = list(range(len(raw)))[::1] 
raw  = raw.iloc[idx]
data = {key: raw[key] for key in raw.columns.unique(0)}
time = data['Time']['Time']
t    = time

data['RFP'] = np.maximum(data['RFP'], 1)
data['GFP'] = np.maximum(data['GFP'], 1)

fig, AX = sim.figure(2, 3, None)
w       = lambda name, func, *states, **kwargs: apply2data(data, name, func, *states, **kwargs)

RFP_OD2Molar   = 1/18.84/30/1e6
Molar2mol_cell = 1/2.4e9/1e3
OD2V           = 3.6e9*1e3 #fl of cell volume per L
V2aa           = 12e8

data = w('RFP/OD', '/', 'RFP', 'OD')
data = w('RFP_M', RFP_OD2Molar, 'RFP/OD')
data = w('RFP_cell', Molar2mol_cell, 'RFP_M')
data = w('Ribo', 6e23, 'RFP_cell')
data = w('R1', 7459+236, 'Ribo')
data = w('R_frac1', 1/4.16e8, 'R1')
data = w('mu', lambda x, y: x.diff()/y.diff().values/x , 'OD', 'Time') 

data = w('V', OD2V, 'OD')
data = w('prot', V2aa, 'V') #aa/L
data = w('R2', 6e23*(7459+236)*RFP_OD2Molar, 'RFP')
data = w('R_frac2', '/', 'R2', 'prot')

data = w('GFP/OD', '/', 'GFP', 'OD')

#Normalize RFP/OD
to_norm = 1 / data['R_frac1']['p1']['0'].values
data    = w('Norm(R_frac1)', to_norm, 'R_frac1', )


colors = [sim.colors['dull orange'], 
          sim.colors['ocean'], 
          sim.colors['cobalt'],
          sim.colors['marigold'],
          sim.colors['violet'],
          sim.colors['coral']
          ]
colors = [c for c in colors for i in range(1)]
labels = list(data['R_frac1'].columns)

for label, color in zip(labels, colors):
    AX[0].plot(t, data['RFP/OD'][label].values, '+', label=label, color=color)
    AX[1].plot(t, data['mu'][label].values,      '+', label=label, color=color)
    AX[2].plot(t, data['OD'][label].values,      '+', label=label, color=color)
    AX[3].plot(t, data['Norm(R_frac1)'][label].values, '+', label=label, color=color)
    AX[4].plot(t, data['GFP/OD'][label].values,  '+', label=label, color=color)
    AX[5].plot(data['mu'][label].values, data['R_frac1'][label].values, '+', label=label, color=color)
    
    AX[0].set_title('RFP/OD')
    AX[1].set_title('mu')
    AX[2].set_title('OD')
    AX[3].set_title('Norm(R_frac1)')
    AX[4].set_title('GFP/OD')
    
AX[1].legend(fontsize=18)
AX[1].set_ylim(0, 0.035)
AX[3].set_ylim(0, 10)

# # scale = dai(data['mu'])/data['R_frac2']
# # scale = scale.loc[150:400].mean().values[0]
# # print(f'The scaling factor required for data is approximately {scale}')

# # scale = dai(M9['mu'])/M9['R_frac2']
# # scale = scale.loc[150:400].mean().values[0]
# # print(f'The scaling factor required for M9 is approximately {scale}')

# # scale = 0.55
# # new_data_R_frac = pd.concat({'R_frac': scale*data['R_frac2']}, axis=1)
# # new_M9_R_frac = pd.concat({'R_frac': scale*M9['R_frac2']}, axis=1)
# # new_data_OD     = pd.concat({'x': data['OD']}, axis=1)
# # new_M9_OD     = pd.concat({'x': M9['OD']}, axis=1)

# # new_df = pd.concat([new_data_R_frac, new_M9_R_frac, new_data_OD, new_M9_OD], axis=1)
# # t_cols = [col + ('Time', ) for col in new_df.columns]
# # df_t   = pd.DataFrame({col:t for col in t_cols})
# # df_t.index = t
# # new_df = pd.concat({'Data': new_df}, axis=1)
# # new_df.columns = new_df.columns.swaplevel(0, 1)
# # new_df.columns = new_df.columns.swaplevel(1, 2)

# # new_df = pd.concat([new_df, df_t], axis=1)
# # new_df = new_df.loc[:370]
# # new_df = new_df[np.all(new_df != 0, axis=1)]
# # print(new_df)

# # new_df.to_csv('data_MCr.csv', index=False)
