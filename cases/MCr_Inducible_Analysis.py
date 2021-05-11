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
import dunlin.simulation    as sim
import dunlin.curvefit      as cf
import dunlin.dataparser    as dp
import dunlin.traceanalysis as ta

add   = lambda x, y: x + y
minus = lambda x, y: x - y
mul   = lambda x, y: x * y
div   = lambda x, y: x / y

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
    else:
        func_ = func
    
    vectors = [data[s] for s in states]
    new     = func_(*vectors, **kwargs)
    result  = join(data, name, new)
    
    return result

def join(data, name, new):
    
    temp   = pd.concat( {name:new}, axis=1 )
    result = data.join(temp)

    return result

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

df = pd.read_excel('data_210416_MCr_raw.xlsx', header=[0, 1], sheet_name='Compiled')

idx  = list(range(50))[::2] + list(range(50, len(df)))[::10]
time = df['Time']['Time']
data = df.iloc[idx]
t = time.values[idx]
data.index= t


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

LB = data[[c for c in data.columns if 'LB' in c[1]]]
colors = sim.palette_types['light'](len(LB['R_frac1'].columns), color=sim.colors['cobalt'])
labels = list(LB['R_frac1'].columns)

for color, label in zip(colors, labels):
    AX[0].plot(t, LB['R_frac1'][label].values, '+', label=label, color=color)
    AX[1].plot(t, LB['mu'][label].values,      '+', label=label, color=color)
    AX[2].plot(t, LB['OD'][label].values,      '+', label=label, color=color)
    AX[3].plot(t, LB['R_frac2'][label].values, '+', label=label, color=color)
    AX[4].plot(t, LB['GFP/OD'][label].values,  '+', label=label, color=color)
    
M9 = data[[c for c in data.columns if 'M9' in c[1]]]
colors = sim.palette_types['light'](len(M9['R_frac1'].columns), color=sim.colors['ocean'])
labels = list(M9['R_frac1'].columns)

for color, label in zip(colors, labels):
    AX[0].plot(t, M9['R_frac1'][label].values, '+', label=label, color=color)
    AX[1].plot(t, M9['mu'][label].values,      '+', label=label, color=color)
    AX[2].plot(t, M9['OD'][label].values,      '+', label=label, color=color)
    AX[3].plot(t, M9['R_frac2'][label].values, '+', label=label, color=color)
    AX[4].plot(t, M9['GFP/OD'][label].values,  '+', label=label, color=color)
    
AX[0].legend(loc='upper right')
AX[1].legend(loc='upper right')
AX[2].legend(loc='upper right')
AX[3].legend(loc='upper right')
AX[4].legend(loc='upper right')
AX[0].set_title('RFP/OD')
AX[1].set_title('mu')
AX[2].set_title('OD')
AX[3].set_title('Ignore this')
AX[4].set_title('GFP/OD')
new_LB_R_frac = pd.concat({'R_frac': 0.55*LB['R_frac2']}, axis=1)
new_M9_R_frac = pd.concat({'R_frac': 0.55*M9['R_frac2']}, axis=1)
new_LB_OD     = pd.concat({'x': LB['OD']}, axis=1)
new_M9_OD     = pd.concat({'x': M9['OD']}, axis=1)

new_df = pd.concat([new_LB_R_frac, new_M9_R_frac, new_LB_OD, new_M9_OD], axis=1)
t_cols = [col + ('Time', ) for col in new_df.columns]
df_t   = pd.DataFrame({col:t for col in t_cols})
df_t.index = t
new_df = pd.concat({'Data': new_df}, axis=1)
new_df.columns = new_df.columns.swaplevel(0, 1)
new_df.columns = new_df.columns.swaplevel(1, 2)

new_df = pd.concat([new_df, df_t], axis=1)
new_df = new_df.loc[:370]
new_df = new_df[np.all(new_df != 0, axis=1)]
print(new_df)

# new_df.to_csv('data_MCr_Inducible.csv', index=False)